"""
Performance monitoring utilities for PDF Summarizer
"""

import time
import streamlit as st
from typing import Dict, Any
from functools import wraps

class Timer:
    """Context manager for timing operations"""
    def __init__(self, operation_name: str, display: bool = True):
        self.operation_name = operation_name
        self.display = display
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.display:
            st.info(f" Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.display:
            st.success(f" {self.operation_name} completed in {self.duration:.2f} seconds")

def time_operation(operation_name: str):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            # If result is a dict, add timing info
            if isinstance(result, dict):
                result['execution_time'] = duration
                result['operation_name'] = operation_name
            
            return result
        return wrapper
    return decorator

class PerformanceMonitor:
    """Track and display performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float, unit: str = "seconds"):
        """Record a performance metric"""
        self.metrics[name] = {
            'value': value,
            'unit': unit,
            'timestamp': time.time()
        }
    
    def display_metrics(self):
        """Display performance metrics in Streamlit"""
        if not self.metrics:
            return
        
        st.subheader("⚡Performance Metrics")
        
        cols = st.columns(len(self.metrics))
        for i, (name, data) in enumerate(self.metrics.items()):
            with cols[i]:
                st.metric(
                    label=name.replace('_', ' ').title(),
                    value=f"{data['value']:.2f} {data['unit']}"
                )
    
    def get_total_time(self) -> float:
        """Get total processing time"""
        return sum(metric['value'] for metric in self.metrics.values() if metric['unit'] == 'seconds') 