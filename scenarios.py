def emergency_spike():
    return [
        {"id": i, "severity": "critical", "needs_ventilator": True, "waiting_time": 0}
        for i in range(10)
    ]

def normal_day():
    return [
        {"id": i, "severity": "medium", "needs_ventilator": False, "waiting_time": 0}
        for i in range(5)
    ]

def mixed_case():
    return [
        {"id": 1, "severity": "critical", "needs_ventilator": True, "waiting_time": 0},
        {"id": 2, "severity": "low", "needs_ventilator": False, "waiting_time": 0},
        {"id": 3, "severity": "medium", "needs_ventilator": False, "waiting_time": 0},
    ]