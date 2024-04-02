from collections import deque
from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class ScheduledNode:
    type: str
    chunk: int
    stage: int
    minibatch: int
    start_time: int
    completion_time: int
    rollback: bool = False


FORWARD = 0
BACKWARD = 1
WEIGHT = 2


class PipelineGraph(object):
    def __init__(
        self, n_stage, n_micro, f_cost, b_cost, w_cost, c_cost,
        f_mem, b_mem, w_mem, max_mem=None,
    ):
        self.n_node = 6 * n_stage * n_micro
        self.n_stage = n_stage
        self.n_micro = n_micro
        self.f_cost = f_cost
        self.b_cost = b_cost
        self.w_cost = w_cost
        self.c_cost = c_cost
        self.f_mem = f_mem
        self.b_mem = b_mem
        self.w_mem = w_mem
        self.fbw_cost = [f_cost, b_cost, w_cost]
        self.fbw_mem = [f_mem, b_mem, w_mem]
        self.max_mem = max_mem or f_mem * self.n_stage * 2

    def get_id(self, cat, chunk, stage, micro):
        return cat * 2 * self.n_stage * self.n_micro + \
            chunk * self.n_stage * self.n_micro + \
            stage * self.n_micro + \
            micro

    def try_v_schedule(self, fill_f=True, fill_b=True, approved_bubble=None):
        # 初始化计数器count,对于每个stage都有6个计数器(对应F/B/W,每个有两个chunk)
        count = []
        for i in range(self.n_stage):
            count.append([0] * 6)

        # 初始化结束时间数组end_time,节点编号从0到n_node-1
        end_time = [-1] * self.n_node
        # 初始化当前时间cur_time为全0
        cur_time = [0] * self.n_stage
        # 初始化内存占用mem为全0
        mem = [0] * self.n_stage
        # 初始化每个stage的泡沫时间为全0
        stage_bubble = [0] * self.n_stage
        # 初始化待处理的W任务队列pending_w,每个stage一个
        pending_w = [deque() for _ in range(self.n_stage)]
        # 初始化调度方案schedule为n_stage个空列表
        schedule = [[] for _ in range(self.n_stage)]
        # 生成n_stage个用于输出的前缀字符串
        stage_str = ["    " * i for i in range(self.n_stage)]

        # 如果approved_bubble为None,则初始化为n_stage个-1
        if approved_bubble is None:
            approved_bubble = [-1] * self.n_stage
        # 计算approved_bubble中的最大值
        max_approved_bubble = max(approved_bubble)

        # 定义获取最大stage泡沫时间的函数
        def get_max_stage_bubble(stage=-1):
            max_stage_bubble = 0
            for bb in stage_bubble:
                max_stage_bubble = max(max_stage_bubble, bb)
            # 如果给定了stage,则还需要考虑该stage的approved_bubble
            if stage >= 0:
                max_stage_bubble = max(
                    max_stage_bubble, max_approved_bubble - approved_bubble[stage])
            return max_stage_bubble

        # 定义处理待处理W任务队列的函数
        def put_w(stage):
            assert len(pending_w[stage]) > 0
            _, chunk_, _ = pending_w[stage].popleft()
            put(2, chunk_, stage)

        # 定义插入F/B/W任务的函数
        def put(cat, chunk, stage, assert_cnt=True):
            """
            @param cat: 任务类型,0为F,1为B,2为W
            @param chunk: 任务块,0为chunk 0,1为chunk 1
            @param stage: stage编号
            @param assert_cnt: 是否检查计数器
            """
            task_end_time = _no_bubble = cur_time[stage] + self.fbw_cost[cat]
            # Note: 为什么是 cat * 2 + chunk?
            # 0 -> F0, 1 -> F1, 2 -> B0, 3 -> B1, 4 -> W0, 5 -> W1
            # 默认 vpp degree 是 2，这里看起来需要后续需要修改为 cat * vpp_degree + chunk
            _cnt = count[stage][cat * 2 + chunk]
            # assert _cnt < self.n_micro
            if _cnt >= self.n_micro:
                if not assert_cnt:
                    stage_str[stage] += "    "
                    cur_time[stage] = task_end_time  # TODO
                    return
                assert False
            assert mem[stage] + self.fbw_mem[cat] <= self.max_mem
            # 更新输出字符串, FfBbWw 分别代表 F/B/W 任务,后面的数字代表任务编号
            stage_str[stage] += "FfBbWw"[cat * 2 + chunk] + \
                str(_cnt + 1) + " " * (3 - len(str(_cnt + 1)))
            
            if cat > 0 or chunk > 0:
                # 如果任务不是前向传播的第一个chunk（即不是F0_1），则需要检查依赖任务是否完成
                last_id = cat * 2 + chunk - 1
                # 计算依赖任务的标识
                if cat < 2:
                    # 如果是前向或反向传播任务，确认其依赖的前一个任务已完成
                    assert end_time[self.get_id(
                        last_id // 2, last_id % 2, stage, _cnt)] >= 0
                else:
                    # 对于权重更新任务，确认其依赖的反向传播任务已完成
                    assert end_time[self.get_id(1, chunk, stage, _cnt)] >= 0
                    
            # ------------------------------------------------
            # 更新当前任务的结束时间 Note(Sonder): 这里也需要适配 vpp_degree > 2 的逻辑
            # ------------------------------------------------ 
            if chunk == 1 and cat < 2:
                # 如果是反向传播的第二个chunk，需要等待下一个stage相应的任务完成
                if stage < self.n_stage - 1:
                    # 获取下一个stage的相应任务ID
                    _fa_id = self.get_id(cat, chunk, stage + 1, _cnt)
                    # 确保下一个stage的任务已完成
                    assert end_time[_fa_id] >= 0
                    # 更新当前任务的预计结束时间，考虑通信成本和任务本身的执行时间
                    # self.c_cost 是通信成本
                    # self.fbw_cost[cat] 是任务本身的执行时间
                    task_end_time = max(task_end_time, end_time[_fa_id] + self.c_cost + self.fbw_cost[cat])
            if chunk == 0 and cat < 2:
                # 如果是前向或反向传播的第一个chunk，需要等待前一个stage相应的任务完成
                if stage > 0:
                    # 获取前一个stage的相应任务ID
                    _fa_id = self.get_id(cat, chunk, stage - 1, _cnt)
                    # 确保前一个stage的任务已完成
                    assert end_time[_fa_id] >= 0, f"{cat}, {chunk}, {stage}, {_cnt}"
                    # 更新当前任务的预计结束时间，考虑通信成本和任务本身的执行时间
                    task_end_time = max(task_end_time, end_time[_fa_id] + self.c_cost + self.fbw_cost[cat])
            _id = self.get_id(cat, chunk, stage, _cnt)
            # 为当前任务生成唯一ID
            if count[stage][0] > 0:
                # 如果在当前stage已经有任务被安排，则计算stage内的空闲时间（泡沫）
                stage_bubble[stage] += task_end_time - _no_bubble
            # 更新当前任务的结束时间
            end_time[_id] = task_end_time
            # 更新当前stage的时间，以反映新任务的安排
            cur_time[stage] = task_end_time
            # 更新内存使用情况
            mem[stage] += self.fbw_mem[cat]
            # 将任务加入到当前stage的调度计划中
            schedule[stage].append((cat, chunk, _cnt))
            if cat == 1:
                # 如果是反向传播任务，将对应的权重更新任务加入待处理队列
                pending_w[stage].append((2, chunk, _cnt))
            # 更新当前stage内指定类型和chunk的任务计数
            count[stage][cat * 2 + chunk] += 1

        # ------------------------------------------------
        # 插入F任务的chunk 0
        # ------------------------------------------------
        for i in range(self.n_stage):
            put(FORWARD, 0, i)
         
        # ------------------------------------------------
        # 从最后一个卡开始,插入F任务的chunk 1
        # 结合 VPP 的图来理解，形状是一个 V 形状
        # ------------------------------------------------
        for i in range(self.n_stage - 1, -1, -1):
            if i == self.n_stage - 1: # 最后一个卡直接插入F任务的chunk 1
                put(FORWARD, 1, i)  # 插入F任务的chunk 1
                continue
            tmp = end_time[self.get_id(0, 1, i + 1, 0)] + self.c_cost
            # 如果 i 卡内存 mem[i] 加上 F 任务的内存占用小于最大内存,且当前时间 cur_time[i] 加上 F 任务的时间小于 tmp,且 F 任务的 chunk 0 数量小于 n_micro
            # 则插入 F 任务的 chunk 0
            while mem[i] + self.fbw_mem[FORWARD] * (2 + i * 2) <= self.max_mem and cur_time[i] + self.fbw_cost[FORWARD] <= tmp and count[i][0] < self.n_micro:
                for j in range(i + 1):
                    put(FORWARD, 0, j)  # 插入F任务的chunk 0
            put(FORWARD, 1, i)  # 插入F任务的chunk 1

        # ------------------------------------------------
        # 插入第一个backward之前剩下的 F
        # 形成 V 形
        # ------------------------------------------------
        iter_chunk_ = 0
        end_tmp = 0
        for i in range(self.n_stage):
            if i == 0:
                end_tmp = cur_time[0] + self.fbw_cost[1]
                continue
            tmp = end_tmp + self.c_cost
            while count[i][0] + count[i][1] < count[i - 1][0] + count[i - 1][1] or count[i][1] <= count[i - 1][1] < self.n_micro:
                for j in range(self.n_stage - 1, i - 1, -1):
                    if count[j][iter_chunk_] < self.n_micro:
                        put(FORWARD, iter_chunk_, j)
                iter_chunk_ = 1 - iter_chunk_

        # ------------------------------------------------
        # 逐步插入B和W任务,尽量填充泡沫
        # ------------------------------------------------
        # Note(sonder): 为什么是 2 * self.n_micro?
        # 因为每个stage有两个 chunk，每个 chunk 最多有 n_micro 个任务
        # 这里后续需要适配 vpp_degree > 2 的逻辑，变成 vpp_degree * self.n_micro
        for _ in range(2 * self.n_micro):
            # 1. 检查内存,如果不够就先处理 pending_w 队列
            for i in range(self.n_stage):
                while mem[i] + self.fbw_mem[BACKWARD] > self.max_mem:
                    assert len(pending_w[i]) > 0
                    put_w(i)
            # Note(sonder): 这里默认也是 vpp_degree = 2，需要后续适配 vpp_degree > 2 的逻辑
            b0_ranks, b1_ranks = [], []

            # 2. 根据条件分别将每个stage插入b0或b1列表
            for i in range(self.n_stage):
                # 如果 B 任务的 chunk 1 数量大于等于 chunk 0 数量,则插入 b0_ranks
                if count[i][3] >= count[i][2]:
                    b0_ranks.append(i)
                elif i == self.n_stage - 1: # 如果是最后一个卡,则插入 b1_ranks
                    b1_ranks.append(i)
                else:
                    fa_id = self.get_id(1, 1, i + 1, count[i][3])
                    if end_time[fa_id] >= 0 or count[i][2] >= self.n_micro:
                        b1_ranks.append(i)
                    else:
                        b0_ranks.append(i)
            b_ranks = [] # B任务列表

            # Node(sonder): 为什么要先加入 b1_ranks 再加入 b0_ranks?
            # 因为 backward 依赖关系和 forward 是相反的，backward 的 chunk 0 依赖 chunk 1
            
            # 3. 先插入b1_ranks中的B任务
            # Note(sonder): 这里是倒序插入，再结合图看一下为啥是倒序插入
            for i in reversed(b1_ranks):
                b_ranks.append((i, 1)) # (stage编号, chunk编号)

            # 4. 再插入b0_ranks中的B任务
            for i in b0_ranks:
                b_ranks.append((i, 0)) # (stage编号, chunk编号)

            # 5. 插入B任务,尽量填充泡沫
            # Note(sonder): 单卡视角下，一次只会插入一个 B 任务 b0/b1
            for i, _chunk_ in b_ranks: 
                fa_id = -1
                if _chunk_ == 1 and i < self.n_stage - 1:
                    fa_id = self.get_id(1, 1, i + 1, count[i][3])
                if _chunk_ == 0 and i > 0:
                    fa_id = self.get_id(1, 0, i - 1, count[i][2])
                # 检查内存,如果不够就先处理pending_w队列
                while len(pending_w[i]) > 0 and fa_id >= 0 and end_time[fa_id] + self.c_cost >= cur_time[i] + self.fbw_cost[2]:
                    # 填充泡沫
                    put_w(i)
                if len(pending_w[i]) > 0 and end_time[fa_id] + self.c_cost - cur_time[i] > get_max_stage_bubble(i) - stage_bubble[i]:
                    # 如果泡沫时间大于0, 则尽量填充泡沫
                    if _chunk_ == 1: # 如果是 chunk 1, 则尽量填充泡沫
                        put_w(i)
                    elif fill_b: # 如果是 chunk 0, 则根据 fill_b 来决定是否填充泡沫
                        put_w(i)
                put(BACKWARD, _chunk_, i) # 插入B任务

            # 6. 插入F任务,尽量填充泡沫
            # Note(sonder): 单卡视角下，一次只会插入一个 F 任务
            for i in range(self.n_stage):
                # 该卡的 F1 都已经插入, 跳过
                if count[i][1] >= self.n_micro:
                    continue
                put_item = None
                # 如果 F1 的数量大于等于 F0 的数量,则插入 F0
                if count[i][1] >= count[i][0]:
                    put_item = 0
                # 如果是最后一个卡,则插入 F1
                elif i == self.n_stage - 1:
                    put_item = 1
                else:
                    if end_time[self.get_id(0, 1, i + 1, count[i][1])] >= 0:
                        put_item = 1
                    elif count[i][0] < self.n_micro:
                        if i == 0:
                            put_item = 0
                        elif end_time[self.get_id(0, 0, i - 1, count[i][0])] >= 0:
                            put_item = 0
                if put_item is None:
                    continue
                # 检查内存,如果不够就先处理pending_w队列
                while mem[i] + self.fbw_mem[FORWARD] > self.max_mem:
                    assert len(pending_w[i]) > 0
                    put_w(i)
                fa_id = -1
                if put_item == 0 and i > 0:
                    fa_id = self.get_id(0, 0, i - 1, count[i][0])
                if put_item == 1 and i < self.n_stage - 1:
                    fa_id = self.get_id(0, 1, i + 1, count[i][1])
                while len(pending_w[i]) > 0 and fa_id >= 0 and end_time[fa_id] + self.c_cost >= cur_time[i] + self.fbw_cost[2]:
                    # 用 w 来填充泡沫
                    put_w(i)
                if len(pending_w[i]) > 0 and end_time[fa_id] + self.c_cost - cur_time[i] > get_max_stage_bubble(i) - stage_bubble[i]:
                    if fill_f:
                        # 用 w 来填充泡沫
                        put_w(i)
                # 插入F任务
                put(FORWARD, put_item, i)

        # 处理剩余的pending_w队列
        for i in range(self.n_stage):
            while len(pending_w[i]) > 0:
                put_w(i)

        # 计算当前调度方案的最大泡沫时间
        max_bubble = get_max_stage_bubble()
        expected_time = sum(self.fbw_cost) * self.n_micro * 2
        bubble_rate = max_bubble / expected_time
        # 如果当前泡沫时间较小,则尝试进一步优化
        if max_approved_bubble < 0 or max_bubble < max_approved_bubble:
            _schedule, _end_time, _max_bubble = self.try_v_schedule(
                fill_f=fill_f, fill_b=fill_b,
                approved_bubble=stage_bubble,
            )
            if _max_bubble < max_bubble:
                return _schedule, _end_time, _max_bubble
        # 返回当前调度方案、结束时间和最大泡沫时间
        for item in stage_str:
            print(item)
        exit(0)
        return schedule, end_time, max_bubble

    def print_details(self, end_time, print_scaling=1):
        for stage in range(self.n_stage):
            stage_str = ['.'] * int(max(end_time) / print_scaling)
            for _cat in range(3):
                for _chunk in range(2):
                    for _micro in range(self.n_micro):
                        _id = self.get_id(_cat, _chunk, stage, _micro)
                        if end_time[_id] < 0:
                            continue
                        end = int(end_time[_id] / print_scaling)
                        start = int(
                            (end_time[_id] - self.fbw_cost[_cat]) / print_scaling)
                        for j in range(start, end):
                            if j == start or j == end - 1:
                                stage_str[j] = "FfBbWw"[_cat * 2 + _chunk]
                            elif j == start + 1:
                                if _micro >= 10:
                                    stage_str[j] = str(_micro // 10)
                                else:
                                    stage_str[j] = str(_micro)
                            elif j == start + 2 and _micro >= 10:
                                stage_str[j] = str(_micro % 10)
                            else:
                                stage_str[j] = "-"
            _str = ""
            for _c in stage_str:
                _str += _c
            print(_str)

    def get_v_schedule(self, only_run_time=False):
        # 初始化调度（计划执行顺序）、结束时间和最大空闲时间（泡沫）变量
        schedule, end_time, max_bubble = None, None, None
        # 根据前向和反向传播的成本以及微批处理数量计算预期时间
        expected_time = sum(self.fbw_cost) * self.n_micro * 2
        # 遍历前向填充和反向填充的所有组合
        for fill_b in [True, False]:
            for fill_f in [True, False]:
                # 尝试生成一个调度计划
                _schedule, _end_time, _max_bubble = self.try_v_schedule(
                    fill_b=fill_b, fill_f=fill_f
                )
                # 如果这是第一个调度或者找到了一个更小的泡沫，则更新调度计划
                if max_bubble is None or _max_bubble < max_bubble:
                    max_bubble = _max_bubble
                    schedule = _schedule
                    end_time = _end_time
        # 如果只需要运行时间，则返回总预期时间加上最大空闲时间
        if only_run_time:
            return max_bubble + expected_time
        # 计算泡沫率，了解调度的效率
        bubble_rate = max_bubble / (expected_time + max_bubble)
        # 打印一些调度的统计信息
        print("%2d %3d, [%5d %5d %5d %5d], %6d -> %6.4f" %
              (self.n_stage, self.n_micro, *self.fbw_cost, self.c_cost, self.max_mem // self.f_mem, bubble_rate))
        # 为每个stage构建详细的执行顺序
        local_order = [[] for _ in range(self.n_stage)]
        # 通信ID字典和计数器，用于管理通信操作的唯一性
        comm_id = {}
        comm_id_counter = 0
        # 初始化后验证时间
        post_validation_time = 0
        # 从最后一个stage开始反向遍历每个stage
        for i in range(self.n_stage - 1, -1, -1):
            # 计算后验证ID
            pv_id = min(2 * (self.n_stage - 1 - i), self.n_micro - 1)
            # 更新后验证时间
            post_validation_time = max(post_validation_time, end_time[self.get_id(
                0, 0, i, pv_id)] - self.fbw_cost[0] - self.c_cost)
            # 遍历发送、接收和无操作，为每个stage添加后验证节点
            for it in ["RECV_", "SEND_", ""]:
                # 跳过特定stage的不必要操作
                if i == 0 and it == "SEND_":
                    continue
                if i == self.n_stage - 1 and it == "RECV_":
                    continue
                # 为当前stage添加后验证节点
                stage_ = i
                local_order[stage_].append(ScheduledNode(
                    type=it + "POST_VALIDATION",
                    chunk=0,
                    stage=stage_,
                    minibatch=0,
                    start_time=post_validation_time,
                    completion_time=post_validation_time,
                ))
                # 更新通信ID
                comm_id[local_order[stage_][-1]] = comm_id_counter
                comm_id_counter += 1
        # 遍历每个stage，根据调度添加计算节点
        for i in range(self.n_stage):
            for _cat_, _chunk_, _micro_ in schedule[i]:
                # 计算完成时间
                complete_time = end_time[self.get_id(
                    _cat_, _chunk_, i, _micro_)]
                # 添加计算节点
                local_order[i].append(ScheduledNode(
                    type="FBW"[_cat_],
                    chunk=_chunk_ if _cat_ == 0 else 1 - _chunk_,
                    stage=i,
                    minibatch=_micro_,
                    start_time=complete_time - self.fbw_cost[_cat_],
                    completion_time=complete_time,
                ))
                # 如果是权重更新（W）则不需要通信
                if _cat_ == 2:  # 没有通信的情况
                    continue
                # 定义前向或反向的通信操作
                cat_str = "FORWARD" if _cat_ == 0 else "BACKWARD"

                def communicate(send_recv, stage_):
                    # 添加通信节点
                    local_order[stage_].append(ScheduledNode(
                        type=send_recv + cat_str,
                        chunk=_chunk_ if _cat_ == 0 else 1 - _chunk_,
                        stage=stage_,
                        minibatch=_micro_,
                        start_time=complete_time,
                        completion_time=complete_time,
                    ))
                    comm_id[local_order[stage_][-1]] = comm_id_counter
                # 根据块的位置和stage管理发送和接收操作
                if _chunk_ == 1 and i > 0:
                    communicate("SEND_", i)
                    communicate("RECV_", i - 1)
                if _chunk_ == 0 and i < self.n_stage - 1:
                    communicate("SEND_", i)
                    communicate("RECV_", i + 1)
                comm_id_counter += 1
        # 对每个stage的节点进行排序，优先处理通信节点
        for rank in range(self.n_stage):
            def even_breaker(x: ScheduledNode):
                # 计算节点总是延迟
                if x.type in ['F', 'B', 'W']:
                    return comm_id_counter
                # 通信节点按它们的唯一通信ID排序
                return comm_id[x]
            local_order[rank] = list(sorted(
                local_order[rank],
                key=lambda x: (x.start_time, even_breaker(x))
            ))
            # 如果接收操作与前一个计算节点重叠，则重新排序以优先执行接收，允许重叠
            for i in range(len(local_order[rank])):
                if i > 0 and local_order[rank][i - 1].type in {'F', 'B', 'W'} and \
                    local_order[rank][i].type.startswith('RECV') and \
                    "POST_VALIDATION" not in local_order[rank][i].type and \
                        local_order[rank][i].start_time <= local_order[rank][i - 1].completion_time:
                    local_order[rank][i], local_order[rank][i -
                                                            1] = local_order[rank][i - 1], local_order[rank][i]
        # 对需要回滚的通信进行处理
        local_order_with_rollback = [[] for _ in range(self.n_stage)]
        for rank in range(self.n_stage):
            rollback_comm = set()
            if rank > 0:
                for node in local_order[rank - 1]:
                    if node.type == "POST_VALIDATION":
                        break
                    if node.type == "SEND_FORWARD":
                        assert node.chunk == 0
                        rollback_comm.add(node.minibatch)
            for node in local_order[rank]:
                if node.type == "RECV_FORWARD" and node.chunk == 0 and node.minibatch in rollback_comm:
                    rollback = True
                    rollback_comm.remove(node.minibatch)
                else:
                    rollback = False
                local_order_with_rollback[rank].append(ScheduledNode(
                    type=node.type,
                    chunk=node.chunk,
                    stage=node.stage,
                    minibatch=node.minibatch,
                    start_time=node.start_time,
                    completion_time=node.completion_time,
                    rollback=rollback,
                ))
            assert len(rollback_comm) == 0
            # 打印最终的调度执行顺序
            for node in local_order_with_rollback[rank]:
                print(f"{node.type}-{node.minibatch}-{int(node.rollback)}", end=', ')
            print()

        # 返回包含可能回滚操作的调度执行顺序
        return local_order_with_rollback


if __name__ == '__main__':
    settings = [
        # p,   n,     f,     b,     w,   c,    h,  a,  l
        (4, 8, 18522, 18086, 9337, 601, 2304, 24, 24),
        # (8, 32, 18513, 18086, 9331, 626, 2304, 24, 24),
        # (8, 64, 18546, 18097, 9321, 762, 2304, 24, 24),
        # (8, 24, 29718, 29444, 19927, 527, 4096, 32, 32),
        # (8, 32, 29802, 29428, 19530, 577, 4096, 32, 32),
        # (8, 64, 29935, 29621, 19388, 535, 4096, 32, 32),
        # (16, 48, 11347, 11248, 8132, 377, 5120, 40, 48),
        # (16, 64, 11307, 11254, 8101, 379, 5120, 40, 48),
        # (16, 128, 11325, 11308, 8109, 378, 5120, 40, 48),
        # (32, 96, 10419, 10207, 7715, 408, 6144, 48, 64),
        # (32, 128, 10408, 10204, 7703, 408, 6144, 48, 64),
        # (32, 256, 10402, 10248, 7698, 460, 6144, 48, 64),
        # (4, 8, 6, 4, 4, 1, 4096, 32, 32),
        # (8, 24, 29444, 29718, 19927, 527, 4096, 32, 32),
        # ( 8, 32, 16099, 16504,  7589,  540, 2304, 24, 16),
        (16, 48, 14407, 14380,  9676, 1610, 4096, 32, 32),
        (16, 64, 14412, 14393,  9688, 1621, 4096, 32, 32),
        (16, 128, 14316, 14306,  9639, 1619, 4096, 32, 32),
        (24, 72,  6763,  6969,  5251,  755, 5120, 40, 48),
        (24, 96,  6783,  6984,  5259,  758, 5120, 40, 48),
        (24, 192, 6785,  6990,  5260,  770, 5120, 40, 48),
        (32,  96, 9458,  9748,  7288,  879, 6144, 48, 64),
        (32, 128, 9469,  9744,  7306,  892, 6144, 48, 64),
        (32, 256, 9447,  9644,  7193,  887, 6144, 48, 64),
    ]
    s = 1024

    # h, a, s = 4096, 32, 1024
    # cost_f, cost_b, cost_w, cost_c = 29718, 29444, 19927, 527
    for p, n, f, b, w, c, h, a, _ in settings:
        mem_f = 34 * h + 5 * a * s
        mem_w = - 32 * h
        mem_b = - mem_w - mem_f
        for m_offset in range(p + 1):
            graph = PipelineGraph(
                n_stage=p,
                n_micro=n,
                f_cost=f,
                b_cost=b,
                w_cost=w,
                c_cost=c,
                f_mem=mem_f,
                b_mem=mem_b,
                w_mem=mem_w,
                max_mem=mem_f * (p * 2 + m_offset),
            )
            graph.get_v_schedule()
            break
