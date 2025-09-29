import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

def post_modify_terrain(cfg):
    cfg.scene.terrain.terrain_generator.size = (20, 20)
    cfg.scene.terrain.terrain_generator.num_rows = 10
    cfg.scene.terrain.terrain_generator.num_cols = 10
    cfg.scene.terrain.terrain_generator.sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion = 1.0,
        ),
    }
    return

def post_modify_terrain_flat(cfg):
    cfg.scene.terrain.terrain_generator.size = (20, 20)
    cfg.scene.terrain.terrain_generator.num_rows = 10
    cfg.scene.terrain.terrain_generator.num_cols = 10
    cfg.scene.terrain.terrain_generator.sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion = 1.0,
        ),
    }
    return


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion = 1.0,
        ),
    },
)

FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion = 1.0,
        ),
    },
)

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)