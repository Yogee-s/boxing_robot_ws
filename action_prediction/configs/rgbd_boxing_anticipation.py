# RGB-D Boxing Action Recognition Configuration (Anticipation & Variable Speed)
# =============================================================================
# OPTIMIZED FOR:
# 1. Prediction Speed (Lower latency inputs)
# 2. Variable FPS (30 vs 60fps) & Variable Punch Speed (4 to 15+ frames)
# 3. Anticipation (Causal modeling)

# -----------------------------------------------------------------------------
# 1. Model Configuration
# -----------------------------------------------------------------------------
model = dict(
    backbone='swin3d_b',        # Base variant: Strong feature extraction
    in_channels=8,              # 8 channels (RGB + Depth + Delta)
    num_classes=8,              
    pretrained=False,           
    dropout=0.5,                
    
    # Causal Mode
    causal=True,
    causal_window=8,            # Matches new 8-frame input
)

# -----------------------------------------------------------------------------
# 2. Data Configuration
# -----------------------------------------------------------------------------
data = dict(
    # ULTRA-FAST 8-Frame Window
    # Reduced from 16 -> 8 for maximum inference speed on Jetson.
    # At 8 frames:
    # - Stride 1: Perfect for 4-frame fast punches.
    # - Stride 3: Covers 24 frames of "real time" (enough for slow punches).
    num_frames=8,              
    
    # STRIDE IS KEY for 8-frame window:
    # [1] = High Speed (Capture fast motion detail)
    # [2] = Normal Context
    # [3] = Extended Context (See further back in time with fewer frames)
    interval_choices=[1, 2, 3], 
    
    # Clip length must track num_frames * max_interval approx
    # But Swin3D uses fixed num_frames. 
    # Dataset loader handles "sampling" 8 frames from these clip lengths.
    clip_len_choices=[12, 16, 24],       
    
    random_window=True,
    use_delta=True,             
    
    crop_size=112,              
    max_depth=4.0,
    ann_file='/home/boxbunny/Desktop/boxing_action_recognition/processed/boxing_rgbd.pkl',
)

# -----------------------------------------------------------------------------
# 3. Training Hyperparameters
# -----------------------------------------------------------------------------
training = dict(
    epochs=300,                 # EXTENDED for overnight training
    batch_size=16,              
    lr=2e-4,                    
    weight_decay=0.05,          
    label_smoothing=0.25,       
    warmup_epochs=10,           # Longer warmup for stability
    
    lr_schedule='cosine',
    min_lr=1e-6,                # Ensure it doesn't decay to zero too early
)

# -----------------------------------------------------------------------------
# 4. Data Augmentation
# -----------------------------------------------------------------------------
augmentation = dict(
    enabled=False,
    
    color_jitter=0.4,
    random_crop_scale=(0.85, 1.0),
    aspect_ratio_range=(0.9, 1.1),
    
    # Motion Augmentation
    temporal_drop_prob=0.2,     
    blur_prob=0.3,              
)

# -----------------------------------------------------------------------------
# 5. Early Stopping & Logging
# -----------------------------------------------------------------------------
early_stopping = dict(
    enabled=False,              # DISABLED: Run full 500 epochs
    monitor='val_acc',
    patience=50,                
    min_delta=0.001,
)

logging = dict(
    log_interval=10,
    save_interval=5,
    save_best=True,
    save_last=True,             # Always save the end state
)
