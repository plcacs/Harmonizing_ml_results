import pyperf
from typing import Optional, List

I_IDLE: int = 1
I_WORK: int = 2
I_HANDLERA: int = 3
I_HANDLERB: int = 4
I_DEVA: int = 5
I_DEVB: int = 6
K_DEV: int = 1000
K_WORK: int = 1001
BUFSIZE: int = 4
BUFSIZE_RANGE = range(BUFSIZE)

class Packet:
    link: Optional['Packet']
    ident: int
    kind: int
    datum: int
    data: List[int]

    def __init__(self, l: Optional['Packet'], i: int, k: int) -> None:
        self.link = l
        self.ident = i
        self.kind = k
        self.datum = 0
        self.data = [0] * BUFSIZE

    def append_to(self, lst: Optional['Packet']) -> 'Packet':
        self.link = None
        if lst is None:
            return self
        else:
            p = lst
            next_pkt = p.link
            while next_pkt is not None:
                p = next_pkt
                next_pkt = p.link
            p.link = self
            return lst

class TaskRec:
    pass

class DeviceTaskRec(TaskRec):
    pending: Optional[Packet]

    def __init__(self) -> None:
        self.pending = None

class IdleTaskRec(TaskRec):
    control: int
    count: int

    def __init__(self) -> None:
        self.control = 1
        self.count = 10000

class HandlerTaskRec(TaskRec):
    work_in: Optional[Packet]
    device_in: Optional[Packet]

    def __init__(self) -> None:
        self.work_in = None
        self.device_in = None

    def workInAdd(self, p: Packet) -> Optional[Packet]:
        self.work_in = p.append_to(self.work_in)
        return self.work_in

    def deviceInAdd(self, p: Packet) -> Optional[Packet]:
        self.device_in = p.append_to(self.device_in)
        return self.device_in

class WorkerTaskRec(TaskRec):
    destination: int
    count: int

    def __init__(self) -> None:
        self.destination = I_HANDLERA
        self.count = 0

class TaskState:
    packet_pending: bool
    task_waiting: bool
    task_holding: bool

    def __init__(self) -> None:
        self.packet_pending = True
        self.task_waiting = False
        self.task_holding = False

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

def trace(a: Any) -> None:
    global layout
    layout -= 1
    if layout <= 0:
        print()
        layout = 50
    print(a, end='')

TASKTABSIZE: int = 10

class TaskWorkArea:
    taskTab: List[Optional['Task']]
    taskList: Optional['Task']
    holdCount: int
    qpktCount: int

    def __init__(self) -> None:
        self.taskTab = [None] * TASKTABSIZE
        self.taskList = None
        self.holdCount = 0
        self.qpktCount = 0

taskWorkArea = TaskWorkArea()

class Task(TaskState):
    link: Optional['Task']
    ident: int
    priority: int
    input: Optional[Packet]
    handle: Any
    last_packet: Optional[Packet]

    def __init__(self, i: int, p: int, w: Optional[Packet], initialState: TaskState, r: Any) -> None:
        super().__init__()
        self.link = taskWorkArea.taskList
        self.ident = i
        self.priority = p
        self.input = w
        self.packet_pending = initialState.isPacketPending()
        self.task_waiting = initialState.isTaskWaiting()
        self.task_holding = initialState.isTaskHolding()
        self.handle = r
        taskWorkArea.taskList = self
        taskWorkArea.taskTab[i] = self
        self.last_packet = None

    def fn(self, pkt: Optional[Packet], r: Any) -> Optional['Task']:
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

    def runTask(self) -> Optional['Task']:
        if self.isWaitingWithPacket():
            msg = self.input
            self.input = msg.link if msg else None
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
    def __init__(self, i: int, p: int, w: Optional[Packet], s: TaskState, r: DeviceTaskRec) -> None:
        super().__init__(i, p, w, s, r)

    def fn(self, pkt: Optional[Packet], r: DeviceTaskRec) -> Optional['Task']:
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
    def __init__(self, i: int, p: int, w: Optional[Packet], s: TaskState, r: HandlerTaskRec) -> None:
        super().__init__(i, p, w, s, r)

    def fn(self, pkt: Optional[Packet], r: HandlerTaskRec) -> Optional['Task']:
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
    def __init__(self, i: int, p: int, w: Optional[Packet], s: TaskState, r: IdleTaskRec) -> None:
        super().__init__(i, 0, w, s, r)

    def fn(self, pkt: Optional[Packet], r: IdleTaskRec) -> Optional['Task']:
        i_rec = r
        assert isinstance(i_rec, IdleTaskRec)
        super().fn(pkt, r)
        i_rec.count -= 1
        if i_rec.count == 0:
            return self.hold()
        elif (i_rec.control & 1) == 0:
            i_rec.control //= 2
            return self.release(I_DEVA)
        else:
            i_rec.control = (i_rec.control // 2) ^ 53256
            return self.release(I_DEVB)

A: int = ord('A')

class WorkTask(Task):
    def __init__(self, i: int, p: int, w: Optional[Packet], s: TaskState, r: WorkerTaskRec) -> None:
        super().__init__(i, p, w, s, r)

    def fn(self, pkt: Optional[Packet], r: WorkerTaskRec) -> Optional['Task']:
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
            pkt.data[i] = (A + w.count) - 1
        return self.qpkt(pkt)

def schedule() -> None:
    t: Optional[Task] = taskWorkArea.taskList
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
        for _ in range(iterations):
            taskWorkArea.holdCount = 0
            taskWorkArea.qpktCount = 0
            IdleTask(I_IDLE, 1, 10000, TaskState().running(), IdleTaskRec())
            wkq: Optional[Packet] = Packet(None, 0, K_WORK)
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
