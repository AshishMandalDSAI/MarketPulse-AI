import sys,time,uuid,shutil,cProfile,pstats
ROOT,ROWS=sys.argv[1],int(sys.argv[2]); sys.path.insert(0,ROOT)
import numpy as np,pandas as pd
rng=np.random.default_rng(7);weeks=208;stores=ROWS//weeks;d=pd.date_range("2022-01-03",periods=weeks,freq="W-MON");n=weeks*stores
df=pd.DataFrame({"Week":np.tile(d,stores),"Store":np.repeat([f"S{i:04d}" for i in range(stores)],weeks),"Region":np.repeat(rng.choice(["North","South","East","West"],stores),weeks)})
for c,s in [("TV_Spend",3000),("Search_Spend",1200),("Social_Spend",800),("Radio_Spend",600)]: df[c]=rng.gamma(2.0,s/2,n).round(2)
df["Price"]=rng.normal(100,5,n).round(2); df["Revenue"]=(20000+2700*np.log1p(df.TV_Spend)-50*df.Price+rng.normal(0,800,n)).round(2)
from src.data_engine.pipeline import profile_and_map
from src.data_engine.analysis_runner import run_full_analysis
from config.settings import dataset_upload_dir,dataset_processed_dir
_,m=profile_and_map(df); did=f"upload_perf{uuid.uuid4().hex[:8]}"
pr=cProfile.Profile(); t0=time.time(); pr.enable(); run_full_analysis(did,df=df,mapping=m); pr.disable(); print("RFA seconds",round(time.time()-t0,1))
pstats.Stats(pr).sort_stats("cumulative").print_stats(25)
shutil.rmtree(dataset_upload_dir(did),ignore_errors=True);shutil.rmtree(dataset_processed_dir(did),ignore_errors=True)
