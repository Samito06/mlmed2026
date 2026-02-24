# ============================================================
# Step 1: Load and inspect a 3D CT scan from LUNA16
# ============================================================

from pathlib import Path
import numpy as np
import SimpleITK as sitk

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

# Root directory of the project (where this file is located)
ROOT_DIR = Path(__file__).resolve().parent

# Directory containing the downloaded subset
SUBSET_DIR = ROOT_DIR / "subset1"

# ------------------------------------------------------------
# Select one CT scan (.mhd file)
# ------------------------------------------------------------

# List all .mhd files in the subset directory
mhd_files = sorted(SUBSET_DIR.glob("*.mhd"))

if len(mhd_files) == 0:
    raise FileNotFoundError(
        f"No .mhd files found in directory: {SUBSET_DIR}"
    )

# Select the first scan for exploration
mhd_path = mhd_files[0]
print(f"Selected CT scan: {mhd_path.name}")

# ------------------------------------------------------------
# Load the CT scan using SimpleITK
# ------------------------------------------------------------

# Read the medical image
itk_image = sitk.ReadImage(str(mhd_path))

# Convert to a NumPy array
# LUNA16 convention: array shape is (z, y, x)
volume = sitk.GetArrayFromImage(itk_image)

# ------------------------------------------------------------
# Extract spatial metadata
# ------------------------------------------------------------

# Voxel spacing in millimetres (x, y, z)
spacing = np.array(itk_image.GetSpacing(), dtype=float)

# Origin of the scan coordinate system (x, y, z)
origin = np.array(itk_image.GetOrigin(), dtype=float)

# Image size according to ITK (x, y, z)
size = np.array(itk_image.GetSize(), dtype=int)

# ------------------------------------------------------------
# Display main information
# ------------------------------------------------------------

print("Volume shape (z, y, x):", volume.shape)
print("Data type:", volume.dtype)
print("ITK image size (x, y, z):", tuple(size))
print("Voxel spacing (mm) (x, y, z):", tuple(spacing))
print("Image origin (x, y, z):", tuple(origin))
print(
    "Intensity range (approx. HU):",
    float(volume.min()),
    "→",
    float(volume.max())
)
# ============================================================
# Step 2: Visualisation of axial, coronal and sagittal slices
# ============================================================

import matplotlib.pyplot as plt

# Central indices in each dimension
z_center = volume.shape[0] // 2
y_center = volume.shape[1] // 2
x_center = volume.shape[2] // 2

# Extract slices in the three anatomical planes
axial_slice = volume[z_center, :, :]        # (y, x)
coronal_slice = volume[:, y_center, :]      # (z, x)
sagittal_slice = volume[:, :, x_center]     # (z, y)

# ------------------------------------------------------------
# Display axial slice
# ------------------------------------------------------------
plt.figure()
plt.title(f"Axial view (z = {z_center})")
plt.imshow(axial_slice, cmap="gray")
plt.axis("off")
plt.show()

# ------------------------------------------------------------
# Display coronal slice
# ------------------------------------------------------------
plt.figure()
plt.title(f"Coronal view (y = {y_center})")
plt.imshow(coronal_slice, cmap="gray")
plt.axis("off")
plt.show()

# ------------------------------------------------------------
# Display sagittal slice
# ------------------------------------------------------------
plt.figure()
plt.title(f"Sagittal view (x = {x_center})")
plt.imshow(sagittal_slice, cmap="gray")
plt.axis("off")
plt.show()
# ============================================================
# Step 3: Load annotations and filter nodules for the CT scan
# ============================================================

import pandas as pd

# Path to the annotations file
annotations_path = ROOT_DIR / "annotations.csv"
annotations = pd.read_csv(annotations_path)

# In LUNA16, 'seriesuid' matches the filename without extension
seriesuid = mhd_path.stem

# Filter annotations for the selected CT scan
scan_annotations = annotations[annotations["seriesuid"] == seriesuid].copy()

print("\n--- Annotations for current scan ---")
print("Series UID:", seriesuid)
print("Number of annotated nodules:", len(scan_annotations))

if len(scan_annotations) > 0:
    # Display a few rows (main columns)
    cols = ["coordX", "coordY", "coordZ", "diameter_mm"]
    print(scan_annotations[cols].head())
else:
    print("No annotated nodules for this scan (this can happen).")

# ============================================================
# Step 4: Find a scan with annotations and display one nodule
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------------------------------------------
# If the current scan has no annotations, find another one
# ------------------------------------------------------------

if len(scan_annotations) == 0:
    print("\nSearching for a scan with at least one annotation...")

    for candidate_mhd in mhd_files:
        candidate_uid = candidate_mhd.stem
        candidate_ann = annotations[annotations["seriesuid"] == candidate_uid]

        if len(candidate_ann) > 0:
            # Load the new scan
            mhd_path = candidate_mhd
            seriesuid = candidate_uid
            scan_annotations = candidate_ann.copy()

            itk_image = sitk.ReadImage(str(mhd_path))
            volume = sitk.GetArrayFromImage(itk_image)

            spacing = np.array(itk_image.GetSpacing(), dtype=float)
            origin = np.array(itk_image.GetOrigin(), dtype=float)

            print("Annotated scan found:", mhd_path.name)
            print("Number of nodules:", len(scan_annotations))
            break

# Safety check
if len(scan_annotations) == 0:
    raise RuntimeError("No annotated scans found in this subset.")

# ------------------------------------------------------------
# Select the first annotated nodule
# ------------------------------------------------------------

nodule = scan_annotations.iloc[0]

world_x = nodule["coordX"]
world_y = nodule["coordY"]
world_z = nodule["coordZ"]
diameter_mm = nodule["diameter_mm"]

print("\nSelected nodule (world coordinates in mm):")
print(f"x={world_x:.2f}, y={world_y:.2f}, z={world_z:.2f}")
print(f"Diameter: {diameter_mm:.2f} mm")

# ------------------------------------------------------------
# Convert world coordinates (mm) to voxel coordinates
# ------------------------------------------------------------

# ITK convention: (x, y, z)
voxel_x = (world_x - origin[0]) / spacing[0]
voxel_y = (world_y - origin[1]) / spacing[1]
voxel_z = (world_z - origin[2]) / spacing[2]

# Convert to integer indices for NumPy array (z, y, x)
x = int(round(voxel_x))
y = int(round(voxel_y))
z = int(round(voxel_z))

print("\nConverted voxel coordinates:")
print(f"x={x}, y={y}, z={z}")

# ------------------------------------------------------------
# Display axial slice with nodule overlay
# ------------------------------------------------------------

slice_axial = volume[z, :, :]

# Approximate radius in pixels (using x spacing)
radius_pixels = (diameter_mm / 2.0) / spacing[0]

fig, ax = plt.subplots()
ax.set_title("Axial slice with annotated nodule")
ax.imshow(slice_axial, cmap="gray")

circle = patches.Circle(
    (x, y),
    radius=radius_pixels,
    fill=False,
    linewidth=2
)
ax.add_patch(circle)

ax.axis("off")
plt.show()

# ============================================================
# Step 5: Extract a 3D patch around a nodule (data preparation)
# ============================================================

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

PATCH_SIZE = 32  # size of the cubic patch (32x32x32)
half = PATCH_SIZE // 2

# ------------------------------------------------------------
# Function to extract a 3D patch centered on (z, y, x)
# ------------------------------------------------------------

def extract_patch(volume, center_z, center_y, center_x, size):
    half = size // 2

    z_min = center_z - half
    z_max = center_z + half
    y_min = center_y - half
    y_max = center_y + half
    x_min = center_x - half
    x_max = center_x + half

    # Check boundaries
    if (
        z_min < 0 or y_min < 0 or x_min < 0 or
        z_max > volume.shape[0] or
        y_max > volume.shape[1] or
        x_max > volume.shape[2]
    ):
        return None

    patch = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    return patch

# ------------------------------------------------------------
# Extract a positive patch (nodule)
# ------------------------------------------------------------

patch_nodule = extract_patch(volume, z, y, x, PATCH_SIZE)

if patch_nodule is None:
    print("Patch goes outside volume boundaries.")
else:
    print("Extracted nodule patch shape:", patch_nodule.shape)

# ============================================================
# Step 6: Dataset preparation (patches + labels)
# ============================================================

import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PATCH_SIZE = 32
HALF = PATCH_SIZE // 2

def clip_and_scale_hu(arr_zyx):
    """Clip CT intensities to lung window and scale to [0, 1]."""
    arr = np.clip(arr_zyx, -1000, 400).astype(np.float32)
    return (arr + 1000.0) / 1400.0

def extract_patch_zyx(volume_zyx, cz, cy, cx, size=PATCH_SIZE):
    """Extract a cubic patch centered at (cz, cy, cx) in (z,y,x) indexing."""
    half = size // 2
    z0, z1 = cz - half, cz + half
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    if z0 < 0 or y0 < 0 or x0 < 0:
        return None
    if z1 > volume_zyx.shape[0] or y1 > volume_zyx.shape[1] or x1 > volume_zyx.shape[2]:
        return None

    return volume_zyx[z0:z1, y0:y1, x0:x1]

def world_to_voxel_xyz(world_xyz, origin_xyz, spacing_xyz):
    """Convert (x,y,z) in mm to voxel coordinates (x,y,z)."""
    return (np.array(world_xyz, dtype=float) - origin_xyz) / spacing_xyz

def far_enough(candidate_zyx, positives_zyx, min_dist=50.0):
    """Check if a candidate center is far enough from all positive centers."""
    cz, cy, cx = candidate_zyx
    for pz, py, px in positives_zyx:
        d = np.sqrt((cz - pz)**2 + (cy - py)**2 + (cx - px)**2)
        if d < min_dist:
            return False
    return True


# ============================================================
# Step 7: Build samples from one subset (positives + negatives)
# ============================================================

# Lecture annotations (déjà fait en Step 3 normalement)
annotations_path = ROOT_DIR / "annotations.csv"
annotations = pd.read_csv(annotations_path)

# Liste des scans dans le subset
mhd_files = sorted(SUBSET_DIR.glob("*.mhd"))

samples = []  # each element: (mhd_path, center_zyx, label)

NEG_PER_POS = 2
MIN_NEG_DIST = 50.0
MAX_NEG_TRIES = 300

print("\nBuilding training samples from subset...")

for mhd_path in mhd_files:
    uid = mhd_path.stem
    ann = annotations[annotations["seriesuid"] == uid]
    if len(ann) == 0:
        continue

    itk_img = sitk.ReadImage(str(mhd_path))
    vol = sitk.GetArrayFromImage(itk_img)              # (z,y,x)
    spacing = np.array(itk_img.GetSpacing(), float)    # (x,y,z)
    origin = np.array(itk_img.GetOrigin(), float)      # (x,y,z)

    # Positive centers in voxel indices (z,y,x)
    pos_centers = []
    for _, row in ann.iterrows():
        world_xyz = (row["coordX"], row["coordY"], row["coordZ"])
        vx, vy, vz = world_to_voxel_xyz(world_xyz, origin, spacing)
        cz, cy, cx = int(round(vz)), int(round(vy)), int(round(vx))
        # Keep only if patch fits
        if extract_patch_zyx(vol, cz, cy, cx) is not None:
            pos_centers.append((cz, cy, cx))
            samples.append((str(mhd_path), (cz, cy, cx), 1))

    if len(pos_centers) == 0:
        continue

    # Generate negatives: random centers far from positives
    nz, ny, nx = vol.shape
    for (czp, cyp, cxp) in pos_centers:
        created = 0
        for _ in range(MAX_NEG_TRIES):
            cz = random.randint(HALF, nz - HALF - 1)
            cy = random.randint(HALF, ny - HALF - 1)
            cx = random.randint(HALF, nx - HALF - 1)
            if far_enough((cz, cy, cx), pos_centers, min_dist=MIN_NEG_DIST):
                if extract_patch_zyx(vol, cz, cy, cx) is not None:
                    samples.append((str(mhd_path), (cz, cy, cx), 0))
                    created += 1
            if created >= NEG_PER_POS:
                break

print("Total samples:", len(samples))
print("Positives:", sum(1 for s in samples if s[2] == 1))
print("Negatives:", sum(1 for s in samples if s[2] == 0))

if len(samples) < 20:
    print("Warning: very small dataset. Model training will be limited.")


# ============================================================
# Step 8: PyTorch Dataset + DataLoader
# ============================================================

class LunaPatchDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mhd_path, (cz, cy, cx), label = self.samples[idx]

        itk_img = sitk.ReadImage(mhd_path)
        vol = sitk.GetArrayFromImage(itk_img)
        patch = extract_patch_zyx(vol, cz, cy, cx)
        patch = clip_and_scale_hu(patch)  # (32,32,32)

        # Torch tensor: (C, D, H, W) = (1, z, y, x)
        x = torch.from_numpy(patch).unsqueeze(0)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y

# Shuffle and split
random.shuffle(samples)
split = int(0.8 * len(samples))
train_samples = samples[:split]
val_samples = samples[split:]

train_ds = LunaPatchDataset(train_samples)
val_ds = LunaPatchDataset(val_samples)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

print("Train samples:", len(train_ds), "| Val samples:", len(val_ds))


# ============================================================
# Step 9: 3D CNN + training loop (simple and stable)
# ============================================================

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)  # logits (no sigmoid here)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    crit = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            losses.append(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()

    return float(np.mean(losses)), correct / total if total > 0 else 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN3D().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    val_loss, val_acc = evaluate(model, val_loader, device)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train_loss={np.mean(train_losses):.4f} | "
        f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
    )
