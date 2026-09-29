#!/usr/bin/env python3
"""
Crowd Sentinel: Real-Time Crowd Safety & Stampede Early-Warning System
Launcher script.
"""
import uvicorn
import os
import sys

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 STARTING CROWD SENTINEL - STAMPEDE EARLY-WARNING SYSTEM")
    print("🔒 Privacy Mode: Zero Biometrics / Spatial Density & Flow Vectors Only")
    print("🌐 Dashboard URL: http://localhost:8000")
    print("=" * 70)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
