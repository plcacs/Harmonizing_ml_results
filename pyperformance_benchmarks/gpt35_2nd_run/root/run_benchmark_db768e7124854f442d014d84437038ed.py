from typing import List, Optional

I_IDLE: int = 1
I_WORK: int = 2
I_HANDLERA: int = 3
I_HANDLERB: int = 4
I_DEVA: int = 5
I_DEVB: int = 6
K_DEV: int = 1000
K_WORK: int = 1001
BUFSIZE: int = 4
BUFSIZE_RANGE: range = range(BUFSIZE)

class Packet:
    def __init__(self, l: Optional['Packet'], i: int, k: int):
        self.link: Optional['Packet'] = l
        self.ident: int = i
        self.kind: int = k
        self.datum: int = 0
        self.data: List[int] = [0] * BUFSIZE

    def append_to(self, lst: Optional['Packet']) -> 'Packet':
        self.link = None
        if lst is None:
            return self
        else:
            p = lst
            next_p = p.link
            while next_p is not None:
                p = next_p
                next_p = p.link
            p.link = self
            return lst

class TaskRec:
    pass

class DeviceTaskRec(TaskRec):
    def __init__(self):
        self.pending: Optional[Packet] = None

class IdleTaskRec(TaskRec):
    def __init__(self):
        self.control: int = 1
        self.count: int = 10000

class HandlerTaskRec(TaskRec):
    def __init__(self):
        self.work_in: Optional[Packet] = None
        self.device_in: Optional[Packet] = None

    def workInAdd(self, p: Packet) -> Optional[Packet]:
        self.work_in = p.append_to(self.work_in)
        return self.work_in

    def deviceInAdd(self, p: Packet) -> Optional[Packet]:
        self.device_in = p.append_to(self.device_in)
        return self.device_in

class WorkerTaskRec(TaskRec):
    def __init__(self):
        self.destination: int = I_HANDLERA
        self.count: int = 0

class TaskState:
    def __init__(self):
        self.packet_pending: bool = True
        self.task_waiting: bool = False
        self.task_holding: bool = False

    def packetPending(self) -> 'TaskState':
        self.packet_pending = True
        self.task_waiting = False
        self.task_holding = False
        return self

    def waiting(self) -> 'TaskState':
        self.packet_pending = False
        self.task_waiting = True
        self.task_holding = False
        return self

    def running(self) -> 'TaskState':
        self.packet_pending = False
        self.task_waiting = False
        self.task_holding = False
        return self

    def waitingWithPacket(self) -> 'TaskState':
        self.packet_pending = True
        self.task_waiting = True
        self.task_holding = False
        return self

    def isPacketPending(self) -> bool:
        return self.packet_pending

    def isTaskWaiting(self) -> bool:
        return self.task_waiting

    def isTaskHolding(self) -> bool:
        return self.task_holding

    def isTaskHoldingOrWaiting(self) -> bool:
        return self.task_holding or (not self.packet_pending and self.task_waiting)

    def isWaitingWithPacket(self) -> bool:
        return self.packet_pending and self.task_waiting and not self.task_holding

tracing: bool = False
layout: int = 0

def trace(a: str) -> None:
    global layout
    layout -= 1
    if layout <= 0:
        print()
        layout = 50
    print(a, end='')

TASKTABSIZE: int = 10

class TaskWorkArea:
    def __init__(self):
        self.taskTab: List[Optional[Task]] = [None] * TASKTABSIZE
        self.taskList: Optional[Task] = None
        self.holdCount: int = 0
        self.qpktCount: int = 0

taskWorkArea: TaskWorkArea = TaskWorkArea()

class Task(TaskState):
    def __init__(self, i: int, p: int, w: Optional[Packet], initialState: TaskState, r: TaskRec):
        self.link: Optional[Task] = taskWorkArea.taskList
        self.ident: int = i
        self.priority: int = p
        self.input: Optional[Packet] = w
        self.packet_pending: bool = initialState.isPacketPending()
        self.task_waiting: bool = initialState.isTaskWaiting()
        self.task_holding: bool = initialState.isTaskHolding()
        self.handle: TaskRec = r
        taskWorkArea.taskList = self
        taskWorkArea.taskTab[i] = self
        self.last_packet: Optional[Packet] = None

    def fn(self, pkt: Optional[Packet], r: TaskRec) -> None:
        self.last_packet = pkt

    def addPacket(self, p: Packet, old: 'Task') -> 'Task':
        if self.input is None:
            self.input = p
            self.packet_pending = True
            if self.priority > old.priority:
                return self
        else:
            p.append_to(self.input)
        return old

    def runTask(self) -> None:
        if self.isWaitingWithPacket():
            msg = self.input
            self.input = msg.link
            if self.input is None:
                self.running()
            else:
                self.packetPending()
        else:
            msg = None
        return self.fn(msg, self.handle)

    def waitTask(self) -> 'Task':
        self.task_waiting = True
        return self

    def hold(self) -> Optional['Task']:
        taskWorkArea.holdCount += 1
        self.task_holding = True
        return self.link

    def release(self, i: int) -> 'Task':
        t = self.findtcb(i)
        t.task_holding = False
        if t.priority > self.priority:
            return t
        else:
            return self

    def qpkt(self, pkt: Packet) -> 'Task':
        t = self.findtcb(pkt.ident)
        taskWorkArea.qpktCount += 1
        pkt.link = None
        pkt.ident = self.ident
        return t.addPacket(pkt, self)

    def findtcb(self, id: int) -> 'Task':
        t = taskWorkArea.taskTab[id]
        if t is None:
            raise Exception(f'Bad task id {id}')
        return t

class DeviceTask(Task):
    def __init__(self, i: int, p: int, w: Optional[Packet], s: TaskState, r: TaskRec):
        super().__init__(i, p, w, s, r)

    def fn(self, pkt: Optional[Packet], r: TaskRec) -> 'Task':
        d = r
        assert isinstance(d, DeviceTaskRec)
        super().fn(pkt, r)
        if pkt is None:
            pkt = d.pending
            if pkt is None:
                return self.waitTask()
            else:
                d.pending = None
                return self.qpkt(pkt)
        else:
            d.pending = pkt
            if tracing:
                trace(pkt.datum)
            return self.hold()

class HandlerTask(Task):
    def __init__(self, i: int, p: int, w: Optional[Packet], s: TaskState, r: TaskRec):
        super().__init__(i, p, w, s, r)

    def fn(self, pkt: Optional[Packet], r: TaskRec) -> 'Task':
        h = r
        assert isinstance(h, HandlerTaskRec)
        super().fn(pkt, r)
        if pkt is not None:
            if pkt.kind == K_WORK:
                h.workInAdd(pkt)
            else:
                h.deviceInAdd(pkt)
        work = h.work_in
        if work is None:
            return self.waitTask()
        count = work.datum
        if count >= BUFSIZE:
            h.work_in = work.link
            return self.qpkt(work)
        dev = h.device_in
        if dev is None:
            return self.waitTask()
        h.device_in = dev.link
        dev.datum = work.data[count]
        work.datum = count + 1
        return self.qpkt(dev)

class IdleTask(Task):
    def __init__(self, i: int, p: int, w: int, s: TaskState, r: TaskRec):
        super().__init__(i, 0, None, s, r)

    def fn(self, pkt: Optional[Packet], r: TaskRec) -> 'Task':
        i = r
        assert isinstance(i, IdleTaskRec)
        super().fn(pkt, r)
        i.count -= 1
        if i.count == 0:
            return self.hold()
        elif i.control & 1 == 0:
            i.control //= 2
            return self.release(I_DEVA)
        else:
            i.control = (i.control // 2) ^ 53256
            return self.release(I_DEVB)

A: int = ord('A')

class WorkTask(Task):
    def __init__(self, i: int, p: int, w: Optional[Packet], s: TaskState, r: TaskRec):
        super().__init__(i, p, w, s, r)

    def fn(self, pkt: Optional[Packet], r: TaskRec) -> 'Task':
        w = r
        assert isinstance(w, WorkerTaskRec)
        super().fn(pkt, r)
        if pkt is None:
            return self.waitTask()
        if w.destination == I_HANDLERA:
            dest = I_HANDLERB
        else:
            dest = I_HANDLERA
        w.destination = dest
        pkt.ident = dest
        pkt.datum = 0
        for i in BUFSIZE_RANGE:
            w.count += 1
            if w.count > 26:
                w.count = 1
            pkt.data[i] = A + w.count - 1
        return self.qpkt(pkt)

def schedule() -> None:
    t = taskWorkArea.taskList
    while t is not None:
        if tracing:
            print('tcb =', t.ident)
        if t.isTaskHoldingOrWaiting():
            t = t.link
        else:
            if tracing:
                trace(chr(ord('0') + t.ident))
            t = t.runTask()

class Richards:
    def run(self, iterations: int) -> bool:
        for i in range(iterations):
            taskWorkArea.holdCount = 0
            taskWorkArea.qpktCount = 0
            IdleTask(I_IDLE, 1, 10000, TaskState().running(), IdleTaskRec())
            wkq = Packet(None, 0, K_WORK)
            wkq = Packet(wkq, 0, K_WORK)
            WorkTask(I_WORK, 1000, wkq, TaskState().waitingWithPacket(), WorkerTaskRec())
            wkq = Packet(None, I_DEVA, K_DEV)
            wkq = Packet(wkq, I_DEVA, K_DEV)
            wkq = Packet(wkq, I_DEVA, K_DEV)
            HandlerTask(I_HANDLERA, 2000, wkq, TaskState().waitingWithPacket(), HandlerTaskRec())
            wkq = Packet(None, I_DEVB, K_DEV)
            wkq = Packet(wkq, I_DEVB, K_DEV)
            wkq = Packet(wkq, I_DEVB, K_DEV)
            HandlerTask(I_HANDLERB, 3000, wkq, TaskState().waitingWithPacket(), HandlerTaskRec())
            wkq = None
            DeviceTask(I_DEVA, 4000, wkq, TaskState().waiting(), DeviceTaskRec())
            DeviceTask(I_DEVB, 5000, wkq, TaskState().waiting(), DeviceTaskRec())
            schedule()
            if taskWorkArea.holdCount == 9297 and taskWorkArea.qpktCount == 23246:
                pass
            else:
                return False
        return True

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'The Richards benchmark, with super()'
    richard = Richards()
    runner.bench_func('richards_super', richard.run, 1)
