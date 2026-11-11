"""make_charts.py (v3) - fresh cohesive charts; OURS highlighted. Titles in captions."""
import json, os, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
from style import apply_style, INK, MUTE
import analysis2 as A
apply_style()
ROOT=Path(__file__).resolve().parents[1]                           # repo root
FIG=str(ROOT/"figures")+"/"; os.makedirs(FIG,exist_ok=True)   # figures go to figures/
C=A.COLOR; P=A.PRETTY
OURS=json.load(open(ROOT/"results"/"ours_final.json"))["OURS"]     # RAS attribution (run_final2.py)
AUD=json.load(open(ROOT/"results"/"audit_aux.json"))               # explainer audit (run_audit2.py)
FEAT=AUD["features"]
# make OURS visually dominant colour
C=dict(C); C["Stacking"]="#0B6E4F"

def hide(ax,*s):
    for k in s: ax.spines[k].set_visible(False)

# 1) ACCURACY -------------------------------------------------------------
def fig_acc():
    order=["Stacking","CatBoost","LightGBM","XGBoost","GradBoost","RandForest","Ridge"]
    fig,ax=plt.subplots(figsize=(3.5,3.0)); y=np.arange(len(order))[::-1]
    for yi,m in zip(y,order):
        ax.barh(yi,A.MAE[m],color=C[m],alpha=0.95 if m=="Stacking" else 0.85,
                edgecolor="#0B3D2E" if m=="Stacking" else "white",lw=1.3 if m=="Stacking" else 0.4,height=0.62,zorder=2)
        gap="" if m=="Stacking" else f"  (+{A.MAE[m]-A.MAE['Stacking']:.2f})"
        ax.text(A.MAE[m]+0.12,yi,f"{A.MAE[m]:.2f}{gap}",va="center",fontsize=7.2,
                color=INK,fontweight="bold" if m=="Stacking" else "normal")
    ax.axvline(A.MAE["Stacking"],color=C["Stacking"],ls="--",lw=1.0,alpha=0.7,zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([P[m] for m in order],fontsize=8)
    ax.set_xlabel("Mean absolute error (W), lower is better"); ax.set_xlim(40,49.6)
    hide(ax,"top","right"); fig.savefig(FIG+"fig_acc.pdf"); plt.close(fig)

# 2) ROBUSTNESS lines (OURS bold) -----------------------------------------
def fig_robust():
    fig,ax=plt.subplots(figsize=(3.5,3.25)); lv=A.LEVELS
    order=["Ridge","Stacking","CatBoost","GradBoost","RandForest","LightGBM","XGBoost"]
    for m in order:
        rel=[100*(v-A.ROB_CURVE[m][0])/A.ROB_CURVE[m][0] for v in A.ROB_CURVE[m]]
        bold=m in ("Stacking","Ridge","XGBoost")
        ax.plot(lv,rel,marker="o",ms=3.4,lw=2.5 if m=="Stacking" else (1.7 if bold else 1.1),
                color=C[m],alpha=1 if bold else 0.6,zorder=5 if m=="Stacking" else 3,label=P[m])
    ax.set_xlabel("Gaussian sensor-noise level (%)"); ax.set_ylabel("MAE increase vs clean (%)")
    ax.set_xticks(lv); ax.margins(x=0.02)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.20),ncol=4,fontsize=6.3,
              columnspacing=1.0,handlelength=1.5,frameon=False)
    hide(ax,"top","right"); fig.subplots_adjust(bottom=0.30)
    fig.savefig(FIG+"fig_robust.pdf"); plt.close(fig)

# 3) EXPLANATION AUDIT: 3 methods x 3 lenses (grouped) --------------------
def fig_audit():
    methods=["TreeSHAP","Permutation","Gain"]; R=AUD["rows"]
    fid=[R[m]["fidelity"] for m in methods]
    # normalise audc to 0-1 within range for comparable bar heights, keep raw in label
    audc=[R[m]["audc"] for m in methods]; amin,amax=min(audc)-0.2,max(audc)+0.1
    audc_n=[(a-amin)/(amax-amin) for a in audc]
    stab=[R[m]["stability"] for m in methods]
    fig,ax=plt.subplots(figsize=(3.5,3.0)); x=np.arange(len(methods)); w=0.26
    cols=["#0B6E4F","#D9822B","#5E4FA2"]
    b1=ax.bar(x-w,fid,w,color=cols[0],label="fidelity (vs reference)")
    b2=ax.bar(x,audc_n,w,color=cols[1],label="faithfulness (deletion)")
    b3=ax.bar(x+w,stab,w,color=cols[2],label="stability (under noise)")
    for xi,a in zip(x,audc): ax.text(xi,(a-amin)/(amax-amin)+0.03,f"{a:.1f}",ha="center",fontsize=6,color=cols[1])
    ax.set_xticks(x); ax.set_xticklabels(methods,fontsize=8)
    ax.set_ylim(0,1.12); ax.set_ylabel("score (higher = better)")
    ax.legend(fontsize=6.6,loc="upper center",bbox_to_anchor=(0.5,1.16),ncol=3,columnspacing=0.7,handlelength=1.0)
    ax.scatter([0],[1.05],marker="*",s=90,color="#111",zorder=6)
    hide(ax,"top","right"); fig.savefig(FIG+"fig_audit.pdf"); plt.close(fig)

# 4) DELETION curves ------------------------------------------------------
def fig_deletion():
    fig,ax=plt.subplots(figsize=(3.5,3.0)); R=AUD["rows"]
    cmap={"TreeSHAP":"#0B6E4F","Permutation":"#D9822B","Gain":"#5E4FA2"}
    for m,c in cmap.items():
        cv=R[m]["curve"]; k=np.arange(len(cv))
        ax.plot(k,cv,marker="o",ms=2.6,lw=2.2 if m=="TreeSHAP" else 1.4,color=c,
                alpha=1 if m=="TreeSHAP" else 0.8,label=m)
    ax.set_xlabel("features removed, most-relevant-first")
    ax.set_ylabel("MAE increase vs intact (W)")
    ax.legend(fontsize=7,loc="lower right"); ax.set_xlim(0,len(R["TreeSHAP"]["curve"])-1)
    hide(ax,"top","right"); fig.savefig(FIG+"fig_deletion.pdf"); plt.close(fig)

# 5) FEATURE IMPORTANCE: reference vs OURS --------------------------------
def fig_importance():
    gt=AUD["gt"]; ours=OURS["importance"]; sh=AUD["rows"]["TreeSHAP"]["imp"]
    top=np.argsort(-np.array(gt))[:6]; feats=[FEAT[i] for i in top]
    fig,ax=plt.subplots(figsize=(3.5,3.0)); y=np.arange(len(feats))[::-1]; h=0.26
    ax.barh(y+h,[gt[i] for i in top],h,color="#9aa0a6",label="physics reference")
    ax.barh(y,[sh[i] for i in top],h,color="#2E8B57",label="CatBoost TreeSHAP")
    ax.barh(y-h,[ours[i] for i in top],h,color="#0B6E4F",label="OURS (combined)")
    ax.set_yticks(y); ax.set_yticklabels(feats,fontsize=8)
    ax.set_xlabel("normalised feature importance (%)"); ax.set_xlim(0,32)
    ax.legend(fontsize=6.8,loc="lower right"); hide(ax,"top","right")
    fig.savefig(FIG+"fig_importance.pdf"); plt.close(fig)

# 6) PARETO 2D (OURS dominant) with inset ---------------------------------
def fig_pareto():
    import matplotlib.lines as mlines
    rob=np.array([A.DELTA20[m] for m in A.MODELS])
    def ms(m): return 230*(1-(A.DELTA20[m]-rob.min())/(rob.max()-rob.min()))+55
    fig,(axL,axR)=plt.subplots(1,2,sharey=True,figsize=(3.5,3.1),
                               gridspec_kw={"width_ratios":[3.1,1.0],"wspace":0.07})
    cluster=["Stacking","CatBoost","LightGBM","XGBoost","GradBoost","RandForest"]
    for m in cluster:
        on=m in A.PARETO
        axL.scatter(A.MAE[m],A.TAU[m],s=ms(m),color=C[m],edgecolor="black" if on else "none",
                    lw=1.7 if on else 0,alpha=1 if on else 0.6,zorder=6 if on else 4)
    on="Ridge" in A.PARETO
    axR.scatter(A.MAE["Ridge"],A.TAU["Ridge"],s=ms("Ridge"),color=C["Ridge"],
                edgecolor="black" if on else "none",lw=1.7 if on else 0,zorder=6)
    # Pareto frontier across the break
    sl=(A.TAU["Ridge"]-A.TAU["Stacking"])/(A.MAE["Ridge"]-A.MAE["Stacking"])
    def fy(x): return A.TAU["Stacking"]+sl*(x-A.MAE["Stacking"])
    axL.plot([A.MAE["Stacking"],42.45],[A.TAU["Stacking"],fy(42.45)],color="#555",ls="--",lw=1.2,zorder=2)
    axR.plot([47.25,A.MAE["Ridge"]],[fy(47.25),A.TAU["Ridge"]],color="#555",ls="--",lw=1.2,zorder=2)
    # labels (each model once)
    lab={"Stacking":(-0.02,0.045,"center","OURS","#0B6E4F","bold"),
         "XGBoost":(0.10,0.040,"left","XGBoost","#444","normal"),
         "CatBoost":(-0.04,-0.040,"right","CatBoost","#444","normal"),
         "LightGBM":(0.05,-0.045,"left","LightGBM","#444","normal"),
         "GradBoost":(0.10,0.005,"left","GradBoost","#444","normal"),
         "RandForest":(0.02,-0.050,"center","RandForest","#444","normal")}
    for m in cluster:
        dx,dy,ha,tx,col,fw=lab[m]
        axL.annotate(tx,(A.MAE[m],A.TAU[m]),(A.MAE[m]+dx,A.TAU[m]+dy),fontsize=6.6,ha=ha,va="center",color=col,fontweight=fw)
    axR.annotate("Ridge",(A.MAE["Ridge"],A.TAU["Ridge"]),(A.MAE["Ridge"],A.TAU["Ridge"]-0.055),
                 fontsize=6.8,ha="center",va="center",fontweight="bold",color="#444")
    axL.set_xlim(40.45,42.55); axR.set_xlim(47.35,48.45); axL.set_ylim(0.45,1.06)
    axL.set_xticks([41,42]); axR.set_xticks([48])
    # broken-axis spines + diagonal marks
    axL.spines["right"].set_visible(False); axR.spines["left"].set_visible(False)
    hide(axL,"top"); axR.spines["top"].set_visible(False); axR.tick_params(left=False)
    d=.018
    for (ax,xx) in [(axL,1),(axR,0)]:
        kw=dict(transform=ax.transAxes,color="#888",clip_on=False,lw=1.0)
        ax.plot((xx-d,xx+d),(-d,+d),**kw); ax.plot((xx-d,xx+d),(1-d,1+d),**kw)
    axL.set_ylabel(r"Explanation fidelity (Kendall-$\tau$)")
    fig.text(0.56,0.015,"MAE (W), lower better",ha="center",fontsize=8)
    leg=[mlines.Line2D([],[],marker="o",ls="none",mec="black",mfc="#ccc",mew=1.5,ms=8,label="Pareto-optimal"),
         mlines.Line2D([],[],marker="o",ls="none",mfc="#bbb",mec="none",ms=11,label="larger = more robust")]
    axL.legend(handles=leg,loc="upper left",fontsize=6.3,frameon=False,handletextpad=0.3,borderpad=0.2)
    fig.subplots_adjust(bottom=0.16,left=0.16,right=0.97,top=0.97)
    fig.savefig(FIG+"fig_pareto.pdf"); plt.close(fig)

# 7) UTS grouped (OURS wins 2 of 3) ---------------------------------------
def fig_uts():
    order=["Stacking","CatBoost","GradBoost","LightGBM","RandForest","XGBoost","Ridge"]
    profs=list(A.PROFILES.keys())
    fig,ax=plt.subplots(figsize=(3.5,3.0)); x=np.arange(len(profs)); w=0.115
    for i,m in enumerate(order):
        vals=[A.UTS[p][m] for p in profs]
        bars=ax.bar(x+(i-3)*w,vals,w,color=C[m],edgecolor="#0B3D2E" if m=="Stacking" else "white",
                    lw=0.8 if m=="Stacking" else 0.3,label=P[m])
        for p,b in zip(profs,bars):
            if A.WINNER[p]==m: ax.scatter([b.get_x()+b.get_width()/2],[b.get_height()+0.03],marker="*",s=55,color="#111",zorder=6)
    ax.set_xticks(x); ax.set_xticklabels(["Performance","Balanced","Safety"],fontsize=8)
    ax.set_ylabel("Unified Trust Score"); ax.set_ylim(0,0.95)
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,1.17),ncol=4,fontsize=6.3,columnspacing=0.7,handlelength=1.0)
    hide(ax,"top","right"); fig.savefig(FIG+"fig_uts.pdf"); plt.close(fig)

if __name__=="__main__":
    for f in [fig_acc,fig_robust,fig_audit,fig_importance,fig_pareto,fig_uts]:
        f(); print(f.__name__,"ok")
