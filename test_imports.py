#!/usr/bin/env python3
"""Test script to verify all imports work"""

print("Testing imports...")

try:
    from langgraph.graph import StateGraph, END
    print("✅ langgraph.graph imported successfully")
except ImportError as e:
    print(f"❌ Failed to import langgraph.graph: {e}")

try:
    # Try different import patterns for MemorySaver
    try:
        from langgraph.checkpoint.memory import MemorySaver
        print("✅ MemorySaver imported from langgraph.checkpoint.memory")
    except ImportError:
        try:
            from langgraph.checkpoint import MemorySaver
            print("✅ MemorySaver imported from langgraph.checkpoint")
        except ImportError:
            print("⚠️ MemorySaver not found, will use simplified version")
except Exception as e:
    print(f"⚠️ MemorySaver import test: {e}")

try:
    from fastapi import FastAPI
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ Failed to import FastAPI: {e}")

try:
    import streamlit
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Streamlit: {e}")

print("\nInstall missing packages with:")
print("pip install langgraph fastapi uvicorn streamlit")