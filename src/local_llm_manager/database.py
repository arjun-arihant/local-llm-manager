"""SQLite database for benchmark history."""

import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Benchmark result data class."""
    id: Optional[int]
    model_name: str
    timestamp: datetime
    tokens_per_second: float
    prompt_eval_rate: float
    total_duration_ms: int
    load_duration_ms: int
    prompt_eval_count: int
    eval_count: int
    quantize_level: Optional[str]
    hardware_info: Dict[str, Any]


class BenchmarkDB:
    """SQLite database manager for benchmark results."""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default location: ~/.local/share/local-llm-manager/benchmarks.db
            home = Path.home()
            data_dir = home / ".local" / "share" / "local-llm-manager"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "benchmarks.db"
        
        self.db_path = str(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tokens_per_second REAL NOT NULL,
                    prompt_eval_rate REAL NOT NULL,
                    total_duration_ms INTEGER NOT NULL,
                    load_duration_ms INTEGER NOT NULL,
                    prompt_eval_count INTEGER NOT NULL,
                    eval_count INTEGER NOT NULL,
                    quantize_level TEXT,
                    hardware_info TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def save_benchmark(self, result: BenchmarkResult) -> int:
        """Save a benchmark result and return its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO benchmarks 
                (model_name, timestamp, tokens_per_second, prompt_eval_rate, 
                 total_duration_ms, load_duration_ms, prompt_eval_count, eval_count,
                 quantize_level, hardware_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.model_name,
                    result.timestamp.isoformat(),
                    result.tokens_per_second,
                    result.prompt_eval_rate,
                    result.total_duration_ms,
                    result.load_duration_ms,
                    result.prompt_eval_count,
                    result.eval_count,
                    result.quantize_level,
                    json.dumps(result.hardware_info)
                )
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_benchmarks(self, model_name: Optional[str] = None, limit: int = 50) -> List[BenchmarkResult]:
        """Get benchmark results, optionally filtered by model name."""
        with sqlite3.connect(self.db_path) as conn:
            if model_name:
                rows = conn.execute(
                    "SELECT * FROM benchmarks WHERE model_name = ? ORDER BY timestamp DESC LIMIT ?",
                    (model_name, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM benchmarks ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            
            results = []
            for row in rows:
                results.append(BenchmarkResult(
                    id=row[0],
                    model_name=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    tokens_per_second=row[3],
                    prompt_eval_rate=row[4],
                    total_duration_ms=row[5],
                    load_duration_ms=row[6],
                    prompt_eval_count=row[7],
                    eval_count=row[8],
                    quantize_level=row[9],
                    hardware_info=json.loads(row[10])
                ))
            return results
    
    def get_model_stats(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get aggregate statistics for a model."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT 
                    COUNT(*) as count,
                    AVG(tokens_per_second) as avg_tps,
                    MAX(tokens_per_second) as max_tps,
                    MIN(tokens_per_second) as min_tps,
                    AVG(total_duration_ms) as avg_duration
                FROM benchmarks 
                WHERE model_name = ?
                """,
                (model_name,)
            ).fetchone()
            
            if row and row[0] > 0:
                return {
                    "benchmark_count": row[0],
                    "avg_tokens_per_second": round(row[1], 2),
                    "max_tokens_per_second": round(row[2], 2),
                    "min_tokens_per_second": round(row[3], 2),
                    "avg_duration_ms": int(row[4])
                }
            return None
    
    def delete_benchmark(self, benchmark_id: int) -> bool:
        """Delete a benchmark by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM benchmarks WHERE id = ?", (benchmark_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def clear_all(self) -> int:
        """Clear all benchmarks. Returns number of deleted rows."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM benchmarks")
            conn.commit()
            return cursor.rowcount
