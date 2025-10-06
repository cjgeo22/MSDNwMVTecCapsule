# Paste the following inside data/dataset_catalog.py to register 'MVTecCapsule'.
# Search for where other datasets (e.g., 'KSDD2') are defined and mirror their structure.

# --- start of snippet ---
DATASETS['MVTecCapsule'] = {
    'name': 'MVTec-AD capsule (mixed / fully)',
    # When you pass --DATASET_PATH=/data/MVTEC_CAPSULE this will be joined with these subdirs:
    'images_dir': 'images',
    'masks_dir': 'masks',

    # Use the split files we wrote with scripts/build_capsule_dataset.py
    'splits': {
        'train': 'splits/MVTecCapsule/train.txt',
        'segmented_train': 'splits/MVTecCapsule/segmented_train.txt',
        'test': 'splits/MVTecCapsule/test.txt',
    },

    # Optional flags used by loaders in this repo; mirror KSDD2 settings if present:
    'has_pixel_masks': True,
    'paths_are_absolute': True,
    'infer_label_from_path': True,
    'label_keywords': {'ok':'good', 'defect':'bad'},
}
# --- end of snippet ---
