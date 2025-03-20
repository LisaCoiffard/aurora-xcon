from tasks.bd_extractors import (
    get_bumper_contacts,
    get_feet_contact_proportion,
    get_final_xy_position,
    get_mean_laser_measures,
    get_random_descriptor,
)

task_behavior_descriptor_extractor = {
    "half_cheetah": {
        "function": get_feet_contact_proportion,
        "args": {"feet_name": "all"},
    },
    "walker": {
        "function": get_feet_contact_proportion,
        "args": {"feet_name": "all"},
    },
    "ant_maze": {
        "function": get_final_xy_position,
        "args": {},
    },
    "kheperax": {
        "function": get_final_xy_position,
        "args": {},
    },
}

behavior_descriptor_extractor = {
    "xy_pos": {
        "function": get_final_xy_position,
        "args": {},
    },
    "bumper_contacts": {
        "function": get_bumper_contacts,
        "args": {},
    },
    "laser_measures": {
        "function": get_mean_laser_measures,
        "args": {},
    },
    "random": {
        "function": get_random_descriptor,
        "args": {},
    },
    "back_feet": {
        "function": get_feet_contact_proportion,
        "args": {"feet_name": "back"},
    },
    "feet_contact": {
        "function": get_feet_contact_proportion,
        "args": {"feet_name": "all"},
    },
}
