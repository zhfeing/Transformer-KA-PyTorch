from typing import List


def delete_instances(instances: List[int], *all_lists):
    instances = sorted(instances, reverse=True)
    for idx in instances:
        for sub_list in all_lists:
            del sub_list[idx]
