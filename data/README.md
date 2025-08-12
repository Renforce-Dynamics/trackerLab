# Data dir folder arrangement

- `amass`: contain the amass data;
- `configs`: contain the grouped motion configurations;
- `pkl_buffer`: buffer for store the saved grouped motions;
- `retarget_cfg`: store the retarget config;
- `retargeted`: store the retargeted results;
- `skeleton`: skeleton info in json;
    - `templates`: source xml file which could be used for tpose and skeleton generation;
- `tpose`: store the compiled tpose data;
- `motion_align`: cfg for align robot joints with the motion, used for generate joint dof pos;