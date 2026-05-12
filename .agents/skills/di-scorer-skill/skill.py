def di_scorer(accuracy, latency, peak_ram, completion_rate):
    w1, w2, w3, w4 = 0.35, 0.25, 0.15, 0.25
    A_base = 85.0
    
    # Normalization
    # Prevent division by zero for latency by ensuring it's at least 1.0 (or >0)
    L_norm = max(latency / 1.0, 1.0)
    
    # M_norm is the fraction of total potential RAM (16GB) used
    M_norm = peak_ram / 16384.0
    
    # Base feasibility is the completion rate
    feasibility = completion_rate
    
    # Penalty if peak ram exceeds threshold
    if peak_ram > 15500:
        feasibility *= 0.5
        
    # DI formula
    # (w1 * normalized accuracy) + (w2 * inverse normalized latency) 
    # + (w3 * inverse normalized memory constraint) + (w4 * feasibility index)
    
    # Safe bounds to prevent div by zero if M_norm == 0
    inv_M_norm = (1 / M_norm) if M_norm > 0 else 1.0
    
    di = (w1 * (accuracy / A_base)) + (w2 * (1 / L_norm)) + (w3 * inv_M_norm) + (w4 * feasibility)
    
    return float(round(di, 4))
