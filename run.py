#!/usr/bin/env python3
"""
Crowd Sentinel: Real-Time Crowd Safety & Stampede Early-Warning System
Launcher script.
"""
import uvicorn
import os
import sys
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # If 8000 is occupied and port was default, try 8001 or fallback cleanly
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', port)) == 0 and "PORT" not in os.environ:
            port = 8001

    print("=" * 70)
    print("🚀 STARTING CROWD SENTINEL - STAMPEDE EARLY-WARNING SYSTEM")
    print("🔒 Privacy Mode: Zero Biometrics / Spatial Density & Flow Vectors Only")
    print(f"🌐 Dashboard URL: http://localhost:{port}")
    print("=" * 70)
    
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

