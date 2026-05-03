from .mesh import MeshInfo, register_meshes

DENSEPOSE_MESHES_DIR = "/home/camposadmin/Documents/lvis/meshes/"

MESHES = [
    MeshInfo(
        name="cat_7466",
        data="cat_7466.pkl",
        geodists="geodists/geodists_cat_7466.pkl",
    ),
    MeshInfo(
        name="dog_7466",
        data="dog_7466.pkl",
        geodists="geodists/geodists_dog_7466.pkl",
    ),
    MeshInfo(
        name="sheep_5004",
        data="sheep_5004.pkl",
        geodists="geodists/geodists_sheep_5004.pkl",
    ),
    MeshInfo(
        name="zebra_5002",
        data="zebra_5002.pkl",
        geodists="geodists/geodists_zebra_5002.pkl",
    ),
    MeshInfo(
        name="horse_5004",
        data="horse_5004.pkl",
        geodists="geodists/geodists_horse_5004.pkl",
    ),
    MeshInfo(
        name="giraffe_5002",
        data="giraffe_5002.pkl",
        geodists="geodists/geodists_giraffe_5002.pkl",
    ),
    MeshInfo(
        name="elephant_5002",
        data="elephant_5002.pkl",
        geodists="geodists/geodists_elephant_5002.pkl",
    ),
    MeshInfo(
        name="cow_5002",
        data="cow_5002.pkl",
        geodists="geodists/geodists_cow_5002.pkl",
    ),
    MeshInfo(
        name="bear_4936",
        data="bear_4936.pkl",
        geodists="geodists/geodists_bear_4936.pkl",
    ),
    MeshInfo(
        name="smpl_27554",
        data="smpl_27554.pkl",
        geodists="geodists/geodists_smpl_27554.pkl",
    ),
]

register_meshes(MESHES, DENSEPOSE_MESHES_DIR)
