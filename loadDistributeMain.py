from Funcs.LD.loadDistributeFanbi import loadDistributeFanbi
from Funcs.LD.loadDistributePowerFlowOrg import loadDistributePowerFlowOrg
from Funcs.LD.loadDistributePowerFlowLin import loadDistributePowerFlowLin
from Funcs.LD.loadDistribuePowerFlowOrgComplete import loadDistributePowerFlowOrgComplete
from Funcs.admittanceMatrix import admittanceMatrix
from Funcs.controlParams import Params
from MyProcess import Process
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


par = Params()
par.offline_horizon = 18*60*60
par.v_train_max = 2
par.v_train_min = 1
par.v_tss_max = 2
par.v_tss_min = 1
par.p_in_max = 20
par.p_in_min = -20
par.j_in_max = 20
par.j_in_min = -20
process = Process(params=par)
process.compute_DFs()
process.compute_Ys()
PF_data, load_avg = loadDistributePowerFlowOrgComplete(LD_start=0
                                            ,LD_horizon=process.params.offline_horizon
                                            ,DFs=process.DFs
                                            ,DFs_noStop=process.DFs_noStop
                                            ,sec_interval=1
                                            ,PV_Price=None
                                            ,in_PV=False
                                            ,Ys=process.Ys
                                            ,Ycs=process.Ycs
                                            ,Yrs=process.Yrs
                                            ,params=process.params)


from Funcs.LD.secIntervalAvg import secIntervalAvg
import pandas as pd
for i in [1,15,30, 60, 120]:
    p = secIntervalAvg(secInterval = i, data = load_avg)
    print(f'---------- sec = {i} ----------\n')
    # display(p.sum())
    p.to_csv('/home/gzt/Codes/2STAGE/Data/load_avg_biStart_complete/{}'.format(f'PF_biStart_org_18h_sec{str(i)}.csv', index = False))
