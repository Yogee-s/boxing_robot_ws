#!/usr/bin/env python3
"""
Path utilities for consistent data root handling and recursive file finding.
"""
import os
from pathlib import Path


def find_file_recursive(data_root, filename, subdirs=None):
    """
    Recursively find a file within data_root.
    
    Args:
        data_root: Root directory to search
        filename: Name of file to find (e.g., 'boxing_segments_3d.pkl')
        subdirs: Optional list of subdirectories to prioritize (e.g., ['annotations'])
    
    Returns:
        Full path to file if found, None otherwise
    """
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        return None
    
    # First, try prioritized subdirectories
    if subdirs:
        for subdir in subdirs:
            for root, dirs, files in os.walk(data_root):
                if subdir in root.split(os.sep):
                    filepath = os.path.join(root, filename)
                    if os.path.isfile(filepath):
                        return filepath
    
    # Then search everywhere
    for root, dirs, files in os.walk(data_root):
        if filename in files:
            return os.path.join(root, filename)
    
    return None


def find_dir_recursive(data_root, dirname):
    """
    Recursively find a directory within data_root.
    
    Args:
        data_root: Root directory to search
        dirname: Name of directory to find (e.g., 'annotations')
    
    Returns:
        Full path to directory if found, None otherwise
    """
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        return None
    
    for root, dirs, files in os.walk(data_root):
        if dirname in dirs:
            return os.path.join(root, dirname)
    
    return None


def find_annotations_dir(data_root):
    """Find annotations directory recursively within data_root."""
    return find_dir_recursive(data_root, 'annotations')


def find_pkl_file(data_root, pkl_name='boxing_segments_3d.pkl'):
    """Find PKL file recursively within data_root, prioritizing annotations directories."""
    return find_file_recursive(data_root, pkl_name, subdirs=['annotations'])


def find_label_map(data_root):
    """Find label_map.txt, checking data_root first, then parent, then recursively."""
    data_root = os.path.abspath(data_root)
    
    # Try root first
    label_map = os.path.join(data_root, 'label_map.txt')
    if os.path.isfile(label_map):
        return label_map
    
    # Try parent directory (common when data_root is a subdirectory like /data/yogee/)
    parent_dir = os.path.dirname(data_root)
    label_map = os.path.join(parent_dir, 'label_map.txt')
    if os.path.isfile(label_map):
        return label_map
    
    # Try annotations directories
    annotations_dir = find_annotations_dir(data_root)
    if annotations_dir:
        label_map = os.path.join(annotations_dir, 'label_map.txt')
        if os.path.isfile(label_map):
            return label_map
    
    # Recursive search
    return find_file_recursive(data_root, 'label_map.txt')


def find_intrinsics(data_folder):
    """
    Find intrinsics.json within a specific data folder.
    Searches in order: data_folder root, annotations/, then recursively within data_folder.
    
    Args:
        data_folder: Specific data folder path (e.g., 'data/recordings/zakir')
    
    Returns:
        Full path to intrinsics.json if found, None otherwise
    """
    data_folder = os.path.abspath(data_folder)
    
    # Try data folder root first (most common location)
    intrinsics = os.path.join(data_folder, 'intrinsics.json')
    if os.path.isfile(intrinsics):
        return intrinsics
    
    # Try annotations directory
    annotations_dir = find_annotations_dir(data_folder)
    if annotations_dir:
        intrinsics = os.path.join(annotations_dir, 'intrinsics.json')
        if os.path.isfile(intrinsics):
            return intrinsics
    
    # Recursive search within this folder only (don't search parent directories)
    return find_file_recursive(data_folder, 'intrinsics.json')


def get_data_folders(data_root):
    """
    Get all data folders (folders containing frames/ or annotations/) within data_root.
    
    Returns:
        List of (folder_name, folder_path) tuples
    """
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        return []
    
    folders = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            # Check if it looks like a data folder
            has_frames = os.path.isdir(os.path.join(item_path, 'frames'))
            has_annotations = find_annotations_dir(item_path) is not None
            if has_frames or has_annotations:
                folders.append((item, item_path))
    
    return folders

