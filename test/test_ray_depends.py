import ray
import time


@ray.remote 
class A:
    def __init__(self):
        self.records = []

    def run(self, x):
        t0 = time.time()
        time.sleep(1)
        y = [x[0] + 1, x[1] + 2]
        t1 = time.time()
        self.records.append({"op": "A.run", "start_ts": t0, "end_ts": t1, "duration_s": t1 - t0})
        return y

    def timing(self):
        return list(self.records)

@ray.remote
class B:
    def __init__(self):
        self.records = []

    def run(self, x):
        t0 = time.time()
        time.sleep(2)
        y = [x[0] + 3, x[1] + 4]
        t1 = time.time()
        self.records.append({"op": "B.run", "start_ts": t0, "end_ts": t1, "duration_s": t1 - t0})
        return y

    def timing(self):
        return list(self.records)

@ray.remote
class C:
    def __init__(self):
        self.records = []

    def run(self, x, y):
        t0 = time.time()
        time.sleep(3)
        y = [x[0] + y[0], x[1] + y[1]]
        t1 = time.time()
        self.records.append({"op": "C.run", "start_ts": t0, "end_ts": t1, "duration_s": t1 - t0})
        return y

    def timing(self):
        return list(self.records)

j = [0, 0]

ray.init(ignore_reinit_error=True)

a = A.remote()
b = B.remote()
c = C.remote()

a_ref = a.run.remote(j)
b_ref = b.run.remote(j)
c_ref = c.run.remote(a_ref, b_ref)

print("result:", ray.get(c_ref))

a_t = ray.get(a.timing.remote())[0]
b_t = ray.get(b.timing.remote())[0]
c_t = ray.get(c.timing.remote())[0]

print("A timing:", a_t)
print("B timing:", b_t)
print("C timing:", c_t)

a_b_overlap_s = min(a_t["end_ts"], b_t["end_ts"]) - max(a_t["start_ts"], b_t["start_ts"])
print("A/B overlap (s):", max(0.0, a_b_overlap_s))
print("C starts after both A/B finished:", c_t["start_ts"] >= max(a_t["end_ts"], b_t["end_ts"]))

ray.timeline("test_ray_depends.json")
ray.shutdown()