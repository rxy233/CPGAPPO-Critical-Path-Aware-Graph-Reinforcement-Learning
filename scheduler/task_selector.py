# -*- coding: utf-8 -*-
"""
任务顺序选择器

支持多种任务排序策略，让算法能够自主决定任务处理顺序。
"""
import random
from dataclasses import dataclass


@dataclass
class TaskSelector:
    """
    任务顺序选择器

    mode:
      - "cp"     : 关键路径优先（复用 gs.get_tasks(sort_tasks=True)）
      - "cp_rev" : 关键路径反转（从最不重要开始，与 cp 相反）
      - "fifo"   : 先来先服务 (start_time 升序)
      - "slack"  : 松弛时间优先（越小越紧急）
      - "random" : 随机顺序（可复现）
      - "none"   : 不改变顺序（保持输入顺序）
      - "custom" : 用 callback(task_list)->task_list
    """
    mode: str = "cp"
    seed: int = 0
    callback: object = None  # 可传函数：(gs, slot, tasks)->tasks

    def order(self, gs, slot: int, tasks):
        """
        对任务列表进行排序

        Args:
            gs: GraphScheduler 实例
            slot: 当前时隙
            tasks: 未排序的任务列表 [(user_id, subtask_id), ...]

        Returns:
            排序后的任务列表
        """
        if not tasks:
            return []

        m = (self.mode or "cp").lower()

        if m == "cp":
            # 修复只对传入的 tasks 进行关键路径排序，不重新调用 get_tasks
            # 批量获取优先级信息（性能优化）
            task_infos = gs.get_tasks_priority_info_batch(tasks, slot)

            # 排序优先级：1) slack 越小越急  2) rank_u 越大越关键
            def cp_sort_key(task):
                info = task_infos[task]
                return (info["slack"], -info["rank_u"])

            return sorted(tasks, key=cp_sort_key)

        if m == "cp_rev":
            # 关键路径反转：从最不重要的任务开始
            # 与 "cp" 相反：1) slack 越大越不急  2) rank_u 越小越不重要
            # 批量获取优先级信息（性能优化）
            task_infos = gs.get_tasks_priority_info_batch(tasks, slot)

            def cp_rev_sort_key(task):
                info = task_infos[task]
                return (-info["slack"], info["rank_u"])

            return sorted(tasks, key=cp_rev_sort_key)

        if m == "fifo":
            # start_time 小的先（早ready的先）
            return sorted(tasks, key=lambda t: gs.ts.start_time[t[0]].get(t[1], float("inf")))

        if m == "slack":
            # 按松弛时间排序（批量计算，性能优化）
            task_infos = gs.get_tasks_priority_info_batch(tasks, slot)
            return sorted(tasks, key=lambda t: task_infos[t]["slack"])

        if m == "rank_u":
            # 按关键路径优先级排序（批量计算）
            task_infos = gs.get_tasks_priority_info_batch(tasks, slot)
            return sorted(tasks, key=lambda t: -task_infos[t]["rank_u"])

        if m == "random":
            # 随机顺序（使用 seed 保证可复现）
            rng = random.Random((self.seed + 1) * 1000003 + slot)
            tasks2 = list(tasks)
            rng.shuffle(tasks2)
            return tasks2

        if m == "custom" and self.callback is not None:
            # 自定义排序回调
            return list(self.callback(gs, slot, list(tasks)))

        # "none" 或未知模式：保持原始顺序
        return list(tasks)
