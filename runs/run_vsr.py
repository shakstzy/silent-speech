import sys, os, time, torch
chap, cfg, vid = sys.argv[1:4]
torch.zeros(1,device="cuda")  # init CUDA before heavy imports (else OOM on GB10)
sys.path.insert(0, chap); os.chdir(chap)
import av, numpy as np, torchvision.io
def _rv(fn, pts_unit='sec'):  # torchvision >=0.22 dropped read_video
    with av.open(fn) as c:
        fr=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
    return (torch.from_numpy(np.stack(fr)),None,None)
torchvision.io.read_video=_rv
from pipelines.pipeline import InferencePipeline
t=time.time()
p = InferencePipeline(cfg, device=torch.device("cuda:0"), detector="retinaface", face_track=True)
print("load_s", round(time.time()-t,1))
for i in range(2):
    torch.cuda.synchronize(); t=time.time()
    out = p(vid)
    torch.cuda.synchronize(); print("infer_s", round(time.time()-t,2), "peak_gb", round(torch.cuda.max_memory_allocated()/1e9,2))
print("PRED:", out)
