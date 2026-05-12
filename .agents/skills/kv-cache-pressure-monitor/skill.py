def monitor_kv_pressure(current_tokens, max_context_window, current_ram_mb, system_max_ram_mb):
    context_utilization = current_tokens / max_context_window
    ram_utilization = current_ram_mb / system_max_ram_mb
    
    if context_utilization > 0.85 or ram_utilization > 0.90:
        return {'status': 'CRITICAL', 'action': 'FLUSH_CACHE'}
    else:
        return {'status': 'SAFE', 'action': 'CONTINUE'}
