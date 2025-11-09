"""
Cache Manager for Voice2Sign
Stores processed video results using YouTube video ID for quick retrieval
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


class CacheManager:
    def __init__(self, cache_dir: Path = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, video_id: str) -> str:
        """Generate cache key from video ID"""
        return f"video_{video_id}"
    
    def get_cache_file(self, video_id: str) -> Path:
        """Get path to cache file for video"""
        cache_key = self.get_cache_key(video_id)
        return self.cache_dir / f"{cache_key}.json"
    
    def has_cache(self, video_id: str) -> bool:
        """Check if cache exists for video"""
        cache_file = self.get_cache_file(video_id)
        return cache_file.exists()
    
    def load_cache(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load cached data for video"""
        cache_file = self.get_cache_file(video_id)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            return None
    
    def save_cache(self, video_id: str, data: Dict[str, Any]) -> bool:
        """Save processed data to cache"""
        cache_file = self.get_cache_file(video_id)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Cached results for video {video_id}")
            return True
        except Exception as e:
            print(f"❌ Error saving cache: {e}")
            return False
    
    def save_stage_cache(self, video_id: str, stage: str, data: Dict[str, Any]) -> bool:
        """Save cache for specific stage (download, transcribe, gloss, etc)"""
        stage_cache_file = self.cache_dir / f"video_{video_id}_stage_{stage}.json"
        
        try:
            with open(stage_cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Cached stage '{stage}' for video {video_id}")
            return True
        except Exception as e:
            print(f"❌ Error saving stage cache: {e}")
            return False
    
    def load_stage_cache(self, video_id: str, stage: str) -> Optional[Dict[str, Any]]:
        """Load cache for specific stage"""
        stage_cache_file = self.cache_dir / f"video_{video_id}_stage_{stage}.json"
        
        if not stage_cache_file.exists():
            return None
        
        try:
            with open(stage_cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading stage cache: {e}")
            return None
    
    def has_stage_cache(self, video_id: str, stage: str) -> bool:
        """Check if cache exists for specific stage"""
        stage_cache_file = self.cache_dir / f"video_{video_id}_stage_{stage}.json"
        return stage_cache_file.exists()
    
    def clear_cache(self, video_id: str = None) -> bool:
        """Clear cache for specific video or all videos"""
        try:
            if video_id:
                cache_file = self.get_cache_file(video_id)
                if cache_file.exists():
                    cache_file.unlink()
                    print(f"✅ Cleared cache for {video_id}")
                    return True
            else:
                # Clear all cache
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
                print(f"✅ Cleared all cache")
                return True
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False
    
    def list_cached_videos(self) -> list:
        """List all cached video IDs"""
        cached = []
        for cache_file in self.cache_dir.glob("video_*.json"):
            video_id = cache_file.stem.replace("video_", "")
            cached.append(video_id)
        return cached
    
    def get_cache_info(self, video_id: str) -> Dict[str, Any]:
        """Get info about cached video"""
        cache_file = self.get_cache_file(video_id)
        
        if not cache_file.exists():
            return {"cached": False}
        
        stat = cache_file.stat()
        return {
            "cached": True,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime
        }
