#!/usr/bin/env python3
"""
Benchmark & Pipeline Runner for 100 Cases Data File.
Runs 4 Parallel Worker Processes/Threads (WorkerProcess-1 to WorkerProcess-4)
with PyTorch single-thread CPU isolation & mini-batch vectorization.
"""

from __future__ import annotations

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import queue
import psutil
import platform
import threading
import datetime
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass

try:
    from grievance_anonymization.main import (
        collect_files,
        extract_text,
        detect_language,
        FileRecord,
        analyze_records,
        build_excel,
        build_json,
        DEFAULT_OUTPUT_PATH,
        _NER_DEVICE_STR,
        NER_MODELS,
        HYBRID_KEY,
    )
except ImportError:
    from main import (
        collect_files,
        extract_text,
        detect_language,
        FileRecord,
        analyze_records,
        build_excel,
        build_json,
        DEFAULT_OUTPUT_PATH,
        _NER_DEVICE_STR,
        NER_MODELS,
        HYBRID_KEY,
    )


class PeakMemoryTracker:
    def __init__(self, sample_interval: float = 0.05):
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())
        self.peak_rss_bytes = 0
        self.peak_swap_bytes = 0
        self._running = False
        self._thread = None

    def _sample_loop(self):
        while self._running:
            try:
                mem_info = self.process.memory_info()
                rss = mem_info.rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
                
                swap = psutil.swap_memory().used
                if swap > self.peak_swap_bytes:
                    self.peak_swap_bytes = swap
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def start(self):
        self.peak_rss_bytes = self.process.memory_info().rss
        self.peak_swap_bytes = psutil.swap_memory().used
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        rss = self.process.memory_info().rss
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        return {
            "peak_rss_mb": round(self.peak_rss_bytes / (1024 * 1024), 2),
            "peak_rss_gb": round(self.peak_rss_bytes / (1024 * 1024 * 1024), 4),
            "peak_swap_mb": round(self.peak_swap_bytes / (1024 * 1024), 2),
        }


def get_os_capacity():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpu_info = "N/A (CPU Mode)"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = f"{torch.cuda.get_device_name(0)} ({round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB VRAM)"
    except Exception:
        pass

    return {
        "os_system": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False) or os.cpu_count(),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_total_mb": round(mem.total / (1024**2), 2),
        "ram_available_start_gb": round(mem.available / (1024**3), 2),
        "swap_total_gb": round(swap.total / (1024**3), 2),
        "gpu_hardware": gpu_info,
        "execution_device": _NER_DEVICE_STR,
        "python_version": sys.version.split()[0],
    }


def main():
    start_dt = datetime.datetime.now()
    start_timestamp = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    pid = os.getpid()
    os_info = get_os_capacity()
    
    input_dir = (root_dir / "data" / "generated_100_cases").resolve()
    output_excel = (root_dir / "output" / "pii_ner_report_100_cases.xlsx").resolve()
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    log_file_path = output_excel.parent / "execution_100_cases.log"
    log_txt_path = output_excel.parent / "execution_100_cases_log.txt"
    logs_100_dataset_txt = output_excel.parent / "logs_100_dataset.txt"
    metrics_file_path = output_excel.parent / "performance_metrics_100_cases.json"

    log_lines = []
    log_lock = threading.Lock()

    def log(msg: str):
        with log_lock:
            print(msg, flush=True)
            log_lines.append(msg)

    def ts_now():
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

    log("=" * 140)
    log(" PIPELINE RUNNER & BENCHMARK: 100 CASES DATASET")
    log(f" Start Time: {start_timestamp}")
    log("=" * 140)
    log("\n[OS CAPACITY & SYSTEM HARDWARE INFO]")
    log(f"  OS System / Kernel  : {os_info['os_system']} {os_info['os_release']} ({os_info['architecture']})")
    log(f"  CPU Cores           : {os_info['cpu_count_logical']} Logical / {os_info['cpu_count_physical']} Physical")
    log(f"  Total System Memory : {os_info['ram_total_gb']} GB ({os_info['ram_total_mb']} MB)")
    log(f"  Available Memory    : {os_info['ram_available_start_gb']} GB (at execution start)")
    log(f"  Total Swap Memory   : {os_info['swap_total_gb']} GB")
    log(f"  GPU Hardware Info   : {os_info['gpu_hardware']}")
    log(f"  Model Inference Dev : {os_info['execution_device']}")
    log(f"  Python Environment  : Python {os_info['python_version']}")
    log("-" * 140)

    overall_tracker = PeakMemoryTracker()
    overall_tracker.start()
    overall_start_time = time.time()

    steps_metrics = []

    log("\n▶ [STEP 1] Collecting and Ingesting Case Files...")
    s1_tracker = PeakMemoryTracker()
    s1_tracker.start()
    s1_start = time.time()

    files = collect_files(str(input_dir)) if input_dir.exists() else []
    log(f"  Found {len(files)} files in input directory.")

    records = []
    for fp in files:
        raw, ftype = extract_text(fp)
        if not raw.strip():
            continue
        lang = detect_language(raw)
        records.append(
            FileRecord(
                path=fp,
                filename=os.path.basename(fp),
                file_type=ftype,
                language=lang,
                raw_text=raw,
            )
        )

    s1_duration = time.time() - s1_start
    s1_mem = s1_tracker.stop()
    
    log(f"  Ingested {len(records)} active records successfully.")
    log(f"  [STEP 1 COMPLETE] Time: {s1_duration:.4f} seconds | Peak RAM: {s1_mem['peak_rss_mb']} MB ({s1_mem['peak_rss_gb']} GB) | Peak Swap: {s1_mem['peak_swap_mb']} MB")

    steps_metrics.append({
        "step_name": "Step 1: Document Ingestion",
        "duration_seconds": round(s1_duration, 4),
        "peak_ram_mb": s1_mem["peak_rss_mb"],
        "peak_ram_gb": s1_mem["peak_rss_gb"],
        "peak_swap_mb": s1_mem["peak_swap_mb"],
    })

    NUM_WORKERS = 4
    log(f"\n▶ [STEP 2] Running Parallel Analysis (PII Scan + Threaded Multilingual NER with {NUM_WORKERS} Worker Processes/Threads)...")
    
    task_queue = queue.Queue()
    file_submit_ts = {}
    
    for idx, rec in enumerate(records, 1):
        sub_ts = ts_now()
        file_submit_ts[rec.filename] = sub_ts
        task_queue.put((idx, rec))

    s2_tracker = PeakMemoryTracker()
    s2_tracker.start()
    s2_start = time.time()

    if records:
        analyze_records([records[0]], batch_size=24, queue_size=64)

    per_file_metrics = [None] * len(records)
    active_workers_count = 0
    completed_count = 0
    worker_counter_lock = threading.Lock()

    def worker_loop(worker_num: int):
        nonlocal active_workers_count, completed_count
        worker_name = f"WorkerProcess-{worker_num}"
        
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        while True:
            batch_items = []
            with worker_counter_lock:
                while len(batch_items) < 5 and not task_queue.empty():
                    try:
                        item = task_queue.get_nowait()
                        batch_items.append(item)
                    except queue.Empty:
                        break

            if not batch_items:
                break

            with worker_counter_lock:
                active_workers_count += 1

            batch_recs = [item[1] for item in batch_items]
            
            for idx, rec in batch_items:
                start_ts = ts_now()
                rec._start_ts = start_ts
                rec._start_t = time.time()

            try:
                analyze_records(batch_recs, batch_size=24, queue_size=64)
            except Exception as exc:
                log(f"{ts_now()} {worker_name} [ERROR] Failed analyzing batch: {exc}")

            for idx, rec in batch_items:
                f_end_time = time.time()
                end_ts = ts_now()
                f_duration = f_end_time - rec._start_t
                
                pii_count = len(rec.pii_hits)
                ner_count = sum(len(lres.entities) for lres in rec.line_ners.get(HYBRID_KEY, []))
                line_cnt = len([line for line in rec.raw_text.splitlines() if line.strip()])
                
                with worker_counter_lock:
                    completed_count += 1

                per_file_metrics[idx - 1] = {
                    "task_id": idx,
                    "filename": rec.filename,
                    "file_path": rec.path,
                    "worker_name": worker_name,
                    "lines": line_cnt,
                    "submit_timestamp": file_submit_ts.get(rec.filename, rec._start_ts),
                    "start_timestamp": rec._start_ts,
                    "end_timestamp": end_ts,
                    "duration_seconds": round(f_duration, 4),
                    "pii_hits": pii_count,
                    "ner_entities": ner_count,
                    "status": "SUCCESS",
                }
                
                task_queue.task_done()

            with worker_counter_lock:
                active_workers_count -= 1

    threads = []
    for i in range(1, NUM_WORKERS + 1):
        t = threading.Thread(target=worker_loop, args=(i,), name=f"WorkerProcess-{i}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    s2_duration = time.time() - s2_start
    s2_mem = s2_tracker.stop()

    log(f"\n  [STEP 2 COMPLETE] Time: {s2_duration:.4f} seconds | Peak RAM: {s2_mem['peak_rss_mb']} MB ({s2_mem['peak_rss_gb']} GB) | Peak Swap: {s2_mem['peak_swap_mb']} MB")

    steps_metrics.append({
        "step_name": f"Step 2: Parallel PII Scan + NER Inference ({NUM_WORKERS} Workers)",
        "duration_seconds": round(s2_duration, 4),
        "peak_ram_mb": s2_mem["peak_rss_mb"],
        "peak_ram_gb": s2_mem["peak_rss_gb"],
        "peak_swap_mb": s2_mem["peak_swap_mb"],
    })

    log("\n▶ [STEP 3] Generating Output Reports...")
    s3_tracker = PeakMemoryTracker()
    s3_tracker.start()
    s3_start = time.time()

    build_excel(records, str(output_excel), presidio_available=False)
    build_json(records, str(output_excel))

    s3_duration = time.time() - s3_start
    s3_mem = s3_tracker.stop()

    log(f"  [STEP 3 COMPLETE] Time: {s3_duration:.4f} seconds | Peak RAM: {s3_mem['peak_rss_mb']} MB ({s3_mem['peak_rss_gb']} GB) | Peak Swap: {s3_mem['peak_swap_mb']} MB")

    steps_metrics.append({
        "step_name": "Step 3: Excel & JSON Reports Generation",
        "duration_seconds": round(s3_duration, 4),
        "peak_ram_mb": s3_mem["peak_rss_mb"],
        "peak_ram_gb": s3_mem["peak_rss_gb"],
        "peak_swap_mb": s3_mem["peak_swap_mb"],
    })

    overall_duration = time.time() - overall_start_time
    overall_mem = overall_tracker.stop()
    end_dt = datetime.datetime.now()
    end_timestamp = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    total_pii_hits = sum(len(r.pii_hits) for r in records)
    total_ner_entities = sum(
        sum(len(lres.entities) for lres in r.line_ners.get(HYBRID_KEY, []))
        for r in records
    )

    log("\n" + "=" * 80)
    log(" EXECUTION & PERFORMANCE SUMMARY")
    log("=" * 80)
    log(f" Start Time         : {start_timestamp}")
    log(f" End Time           : {end_timestamp}")
    log(f" Total Elapsed Time : {overall_duration:.4f} seconds")
    log(f" System Peak Memory : {overall_mem['peak_rss_mb']} MB ({overall_mem['peak_rss_gb']} GB RAM)")
    log(f" Total Cases Processed: {len(records)}")
    log(f" Total PII Detections : {total_pii_hits}")
    log(f" Total NER Detections : {total_ner_entities}")
    log("=" * 80)

    log_text_content = "\n".join(log_lines) + "\n"
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(log_text_content)
    with open(log_txt_path, "w", encoding="utf-8") as f:
        f.write(log_text_content)
    with open(logs_100_dataset_txt, "w", encoding="utf-8") as f:
        f.write(log_text_content)


if __name__ == "__main__":
    main()
