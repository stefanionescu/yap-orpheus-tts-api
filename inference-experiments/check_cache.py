#!/usr/bin/env python3
"""
Check what's in the Modal persistent volume cache
"""

import modal
import os

# Define persistent volume
volume = modal.Volume.from_name("tensorrt-workspace", create_if_missing=True)

# Simple image
image = modal.Image.debian_slim().pip_install("huggingface_hub")

app = modal.App("check-cache", image=image)

@app.function(volumes={"/workspace": volume})
def check_cache():
    """Check what's cached in the persistent volume"""
    
    cache_dir = "/workspace/hf_cache"
    
    print(f"📁 Checking cache directory: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        print("❌ Cache directory doesn't exist")
        return
    
    print("\n🗂️  Cache contents:")
    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
            # Check if it's a model cache
            if item.startswith("models--"):
                model_name = item.replace("models--", "").replace("--", "/")
                print(f"      → Model: {model_name}")
                
                # Check snapshots
                snapshots_dir = os.path.join(item_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = os.listdir(snapshots_dir)
                    print(f"      → Snapshots: {len(snapshots)}")
                    for snapshot in snapshots:
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        if os.path.isdir(snapshot_path):
                            files = os.listdir(snapshot_path)
                            print(f"        - {snapshot}: {len(files)} files")
        else:
            print(f"  📄 {item}")
    
    # Check if orpheus model is there
    orpheus_cache = os.path.join(cache_dir, "models--canopylabs--orpheus-3b-0.1-ft")
    if os.path.exists(orpheus_cache):
        print(f"\n✅ Orpheus model cache found!")
        print(f"   Path: {orpheus_cache}")
        
        # Check if download is complete
        snapshots_dir = os.path.join(orpheus_cache, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                files = os.listdir(snapshot_path)
                print(f"   Files in snapshot: {len(files)}")
                for f in files:
                    print(f"     - {f}")
        else:
            print("   ❌ No snapshots found - download incomplete")
    else:
        print("\n❌ Orpheus model not cached yet")

@app.local_entrypoint()
def main():
    check_cache.remote()