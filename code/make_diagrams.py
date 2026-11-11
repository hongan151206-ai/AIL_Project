"""make_diagrams.py (v6) - bigger text, fresh cool palette, clear arrows, no covering."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from style import apply_style, INK, MUTE
apply_style()
import os
FIG=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figures")+os.sep   # works from any cwd
os.makedirs(FIG, exist_ok=True)
# fresh cool palette: blue / teal / purple
BLUE="#2563A8"; TEAL="#2A9D8F"; PURP="#6A4C93"
GREEN="#3A8E5A"; AMBER="#C8841F"; CRIM="#B83C32"; SLATE="#56607A"
# pale container borders + ultra-light fills
PB_D="#A9C4E0"; PB_M="#A6D7CF"; PB_K="#C4B6DC"
PF_D="#F3F8FC"; PF_M="#F1FAF7"; PF_K="#F7F4FB"
# soft pastel header fills (deploy)
H_C="#E2ECF7"; H_A="#E1F4EE"; H_E="#FBEBD6"; H_K="#EDE7F6"; H_S="#F7E1DD"

def rbox(ax,x,y,w,h,fc,ec,r=2.2,lw=1.1,z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc,ec=ec,lw=lw,zorder=z,mutation_aspect=1.0))
def arrow(ax,x1,y1,x2,y2,c="#56607A",lw=1.9,ls="-",ms=15):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=ms,lw=lw,color=c,ls=ls,zorder=9,shrinkA=0,shrinkB=0))

# ============ FIG 1: two-tier architecture, no circle badges, generous spacing ============
def architecture():
    fig,ax=plt.subplots(figsize=(7.0,3.5)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
    def shadow(x,y,w,h,r=2.4):
        ax.add_patch(FancyBboxPatch((x+0.5,y-0.8),w,h,boxstyle=f"round,pad=0,rounding_size={r}",
                     fc="#C7CCD6",ec="none",zorder=1,alpha=0.38,mutation_aspect=1.0))
    def sbox(x,y,w,h,fc,ec,r=2.4,lw=1.3,z=3):
        shadow(x,y,w,h,r); rbox(ax,x,y,w,h,fc,ec,r=r,lw=lw,z=z)
    def stage(x,w,title,sub,ec,fill):
        sbox(x,58,w,18,fill,ec)
        ax.text(x+w/2,70,title,fontsize=7.0,fontweight="bold",ha="center",va="center",color="black",zorder=5)
        ax.text(x+w/2,63.5,sub,fontsize=6.4,ha="center",va="center",color="#3a3a3a",zorder=5)
    def harrow(x1,x2,y,lw=1.2,c=SLATE):
        ax.add_patch(FancyArrowPatch((x1,y),(x2,y),arrowstyle="-|>",mutation_scale=12,lw=lw,color=c,zorder=7,shrinkA=0,shrinkB=0))
    def varrow(x,y1,y2,lw=1.4,c=SLATE):
        ax.add_patch(FancyArrowPatch((x,y1),(x,y2),arrowstyle="-|>",mutation_scale=12,lw=lw,color=c,zorder=7,shrinkA=0,shrinkB=0))
    # ---- tier 1: data and models (left to right) ----
    stage(6,22,"Telemetry","188 flights, 12 feat.",BLUE,PF_D)
    stage(35,24,"Preprocess","leakage-free split",BLUE,PF_D)
    stage(64,24,"Model pool","7 regressors + RAS",TEAL,PF_M)
    harrow(28.3,34.7,67); harrow(59.3,63.7,67)
    # ---- vertical link into tier 2 ----
    varrow(76,57.5,46.5,lw=1.5)
    ax.text(78,54,"every model",fontsize=5.8,ha="left",va="center",color=MUTE,style="italic")
    ax.text(78,50.5,"scored",fontsize=5.8,ha="left",va="center",color=MUTE,style="italic")
    # ---- tier 2: three-axis audit panel ----
    shadow(6,6,72,40,r=2.6); rbox(ax,6,6,72,40,PF_M,PB_M,r=2.6,lw=1.5,z=2)
    rbox(ax,6,38,72,8,H_A,PB_M,r=2.6,lw=0,z=3); ax.add_patch(Rectangle((6,38),72,2.2,facecolor=PF_M,zorder=3))
    ax.text(42,42,"Three-Axis Audit",fontsize=6.3,fontweight="bold",ha="center",va="center",color="black",zorder=5)
    chips=[("Accuracy","MAE (W)","lower better",BLUE),
           ("Fidelity",r"Kendall $\tau$","vs reference",TEAL),
           ("Robustness",r"$\Delta$MAE","under noise",AMBER)]
    cw=20.0; gap=2.0; x0=6+(72-3*cw-2*gap)/2
    for k,(t,m,d,c) in enumerate(chips):
        bx=x0+k*(cw+gap)
        shadow(bx,10,cw,24,r=2.0); rbox(ax,bx,10,cw,24,"white",c,r=2.0,lw=1.4,z=4)
        ax.text(bx+cw/2,30.4,t,fontsize=5.6,fontweight="bold",ha="center",va="center",color="black",zorder=6)
        ax.add_patch(FancyArrowPatch((bx+cw*0.30,27.6),(bx+cw*0.70,27.6),arrowstyle="-",lw=1.6,color=c,zorder=6))
        ax.text(bx+cw/2,21.3,m,fontsize=6.0,fontweight="bold",ha="center",va="center",color="#1a1a1a",zorder=6)
        ax.add_patch(FancyArrowPatch((bx+4,16.8),(bx+cw-4,16.8),arrowstyle="-",lw=0.7,color="#D4DBDA",zorder=5))
        ax.text(bx+cw/2,13.3,d,fontsize=6.0,ha="center",va="center",color="#555",style="italic",zorder=6)
    # ---- decision ----
    sbox(82,14,15,24,PF_K,PURP)
    ax.text(89.5,31.5,"Decision",fontsize=7.0,fontweight="bold",ha="center",va="center",color="black",zorder=5)
    ax.text(89.5,24.0,"Trust Score",fontsize=6.4,ha="center",va="center",color="#3a3a3a",zorder=5)
    ax.text(89.5,19.5,"+ Pareto",fontsize=6.4,ha="center",va="center",color="#3a3a3a",zorder=5)
    harrow(78.3,81.7,26)
    fig.savefig(FIG+"fig_architecture.pdf",bbox_inches="tight",pad_inches=0.04); plt.close(fig)

# ============ FIG 8: deployment (clear response path Assistant -> Client) ============
def deployment():
    fig,ax=plt.subplots(figsize=(7.1,3.5)); ax.set_xlim(0,100); ax.set_ylim(0,54); ax.axis("off")
    stages=[("Client",BLUE,H_C,["mission context","telemetry"]),
            ("REST API",GREEN,H_A,["Flask service","predict endpoint"]),
            ("Ensemble",AMBER,H_E,["OURS blend","power (W)"]),
            ("Consensus",PURP,H_K,["TreeSHAP","attribution"]),
            ("Assistant",CRIM,H_S,["LLM rationale","operator-facing"])]
    w=13.8; gap=6.8; x0=2.5; yb=15; h=15; cx=[]
    for i,(title,col,hfc,lines) in enumerate(stages):
        x=x0+i*(w+gap); cx.append(x+w/2)
        rbox(ax,x,yb,w,h,"white",col,r=2.0,lw=1.3)
        rbox(ax,x,yb+h-4.6,w,4.6,hfc,col,r=2.0,lw=0,z=3)
        ax.add_patch(Rectangle((x,yb+h-4.6),w,1.0,facecolor="white",zorder=3))
        ax.text(x+w/2,yb+h-2.3,title,fontsize=7.0,fontweight="bold",ha="center",va="center",color="black",zorder=4)
        for j,ln in enumerate(lines):
            ax.text(x+w/2,yb+6.4-j*3.4,ln,fontsize=6.3,ha="center",color="#2b2b2b",zorder=4)
    for i in range(4):
        xa=cx[i]+w/2; xb=cx[i+1]-w/2
        arrow(ax,xa+0.4,yb+h/2-1.0,xb-0.4,yb+h/2-1.0,ms=13)
        ax.text((xa+xb)/2,yb+h/2+1.7,["request","predict","explain","narrate"][i],fontsize=5.4,ha="center",va="center",color=MUTE,style="italic")
    # offline audit
    rbox(ax,cx[3]-16.5,0.6,33,7.4,"#FBEEEC",CRIM,r=1.8,lw=1.0)
    ax.text(cx[3],5.7,"Offline faithfulness audit",fontsize=6.0,fontweight="bold",ha="center",color="black")
    ax.text(cx[3],2.9,"selects the explainer served online",fontsize=5.7,ha="center",color="#444")
    arrow(ax,cx[3],8.0,cx[3],yb-0.4,c=CRIM,lw=1.3,ms=12)
    ax.text(cx[3]+1.3,(8.0+yb)/2,"vetted",fontsize=5.9,ha="left",color=CRIM,style="italic")
    # ---- response feedback path: Assistant top -> up -> left -> down into Client ----
    PV="#7C5BC0"; ytop=yb+h+6.0
    # riser out of Assistant (solid stub + dotted up) with junction dot
    ax.add_patch(FancyArrowPatch((cx[4],yb+h+0.2),(cx[4],ytop),arrowstyle="-",lw=1.5,color=PV,ls=(0,(2,2)),zorder=8))
    ax.add_patch(Circle((cx[4],yb+h+0.2),0.7,color=PV,zorder=10))   # touch point on Assistant
    ax.add_patch(Circle((cx[4],ytop),0.7,color=PV,zorder=10))       # top-right corner
    ax.text(cx[4]+1.4,yb+h+3.0,"from Assistant",fontsize=5.3,ha="left",va="center",color=PV,style="italic")
    # horizontal dashed run to above Client
    ax.add_patch(FancyArrowPatch((cx[4],ytop),(cx[0],ytop),arrowstyle="-",lw=1.5,color=PV,ls=(0,(5,2.5)),zorder=8))
    ax.add_patch(Circle((cx[0],ytop),0.7,color=PV,zorder=10))       # top-left corner
    # drop down into Client (arrowhead onto block)
    arrow(ax,cx[0],ytop,cx[0],yb+h+0.4,c=PV,lw=1.5,ms=13)
    ax.text((cx[0]+cx[4])/2,ytop+1.9,"response:  power (W)  +  attribution  +  rationale",fontsize=7.0,ha="center",color="#5B3FA0",fontweight="bold")
    fig.savefig(FIG+"fig_deploy.pdf"); plt.close(fig)

# ============ FIG 9: decision tool (unchanged structure, cool palette tweak) ============
def tool():
    fig,ax=plt.subplots(figsize=(7.1,3.3)); ax.set_xlim(0,100); ax.set_ylim(0,48); ax.axis("off")
    rbox(ax,1,1,98,46,"#FCFCFD","#CBCFD6",r=2.0,lw=1.2,z=1)
    rbox(ax,1,42,98,5,H_C,BLUE,r=2.0,lw=0,z=2); ax.add_patch(Rectangle((1,42),98,1.4,facecolor="#FCFCFD",zorder=2))
    ax.text(4,44.5,"TRUST-UAV  \u00b7  multi-criteria deployment decision support",fontsize=8.2,color="black",fontweight="bold",va="center")
    ax.text(96,44.5,"v3.0",fontsize=7.4,color=BLUE,va="center",ha="right")
    ax.text(4,39,"MISSION PROFILE",fontsize=7.3,fontweight="bold",color="#333")
    for i,(c,v,col) in enumerate([("Accuracy",0.60,BLUE),("Fidelity",0.20,TEAL),("Robustness",0.20,AMBER)]):
        yy=35-i*4.5
        ax.text(4,yy+0.8,c,fontsize=6.9,color="#333")
        ax.add_patch(Rectangle((4,yy-1.4),20,0.8,facecolor="#E5E7EB")); ax.add_patch(Rectangle((4,yy-1.4),20*v,0.8,facecolor=col))
        ax.add_patch(Circle((4+20*v,yy-1.0),0.8,color=col,zorder=4)); ax.text(25.5,yy-0.5,f"w={v:.2f}",fontsize=6.3,color=MUTE,va="center")
    ax.text(4,19,"CONSTRAINTS",fontsize=7.3,fontweight="bold",color="#333")
    for i,(c,on) in enumerate([("Regulatory certification",False),("Real-time inference",True),("Sensor drift expected",True)]):
        yy=15.4-i*3.3
        ax.add_patch(Rectangle((4,yy-0.85),1.3,1.3,facecolor=BLUE if on else "white",ec="#888",lw=0.8))
        if on: ax.text(4.65,yy-0.2,"x",fontsize=6.0,color="white",ha="center",va="center",fontweight="bold")
        ax.text(6.2,yy-0.15,c,fontsize=6.3,color="#333")
    rbox(ax,33,4.5,28,34,"#F2F7FB",BLUE,r=2.0,lw=1.0,z=2)
    ax.text(47,35.6,"RECOMMENDED",fontsize=7.2,fontweight="bold",color=BLUE,ha="center")
    ax.text(47,31.0,"OURS",fontsize=15,fontweight="bold",color=INK,ha="center")
    ax.text(47,27.2,"Unified Trust Score  0.80",fontsize=6.8,color=MUTE,ha="center")
    for i,(lab,val) in enumerate([("MAE","40.8 W"),(r"$\tau$","0.64"),(r"$\Delta$@20%","+14%")]):
        xx=39+i*8.0
        ax.text(xx,23.2,val,fontsize=6.8,fontweight="bold",ha="center"); ax.text(xx,21.1,lab,fontsize=5.9,color=MUTE,ha="center")
    ax.plot([35.5,58.5],[19.2,19.2],color="#D6E3E3",lw=0.8)
    ax.text(47,17.2,"Rationale",fontsize=6.7,fontweight="bold",ha="center",color="#333")
    ax.text(47,15.2,"Most accurate model;\nPareto-dominates every baseline.\nConsensus flags payload as\nthe dominant power driver.",
            fontsize=5.7,ha="center",va="top",color="#444",linespacing=1.45)
    ax.text(64,39,"MODEL RANKING",fontsize=7.3,fontweight="bold",color="#333")
    ax.text(64,36.4,"profile: performance-critical",fontsize=6.1,color=MUTE,style="italic")
    rank=[("OURS",0.80,BLUE,True),("CatBoost",0.76,TEAL,False),("GradBoost",0.69,"#7FB069",False),
          ("LightGBM",0.69,AMBER,False),("RandForest",0.62,"#B08968",False),("XGBoost",0.61,CRIM,False),("Ridge",0.40,PURP,True)]
    for i,(nm,sc,col,par) in enumerate(rank):
        yy=32.2-i*4.0
        ax.text(64.2,yy,f"{i+1}",fontsize=6.8,color="#888",ha="left")
        ax.text(66.6,yy,nm,fontsize=6.6,color="#222",va="center",fontweight="bold" if nm=="OURS" else "normal")
        if par:
            ax.add_patch(Circle((78.0,yy),0.75,fc="#E6F0F7",ec=BLUE,lw=0.6,zorder=4)); ax.text(78.0,yy,"P",fontsize=4.8,color=BLUE,ha="center",va="center",zorder=5)
        ax.add_patch(Rectangle((80.5,yy-0.85),13.0,1.5,facecolor="#EAECEF")); ax.add_patch(Rectangle((80.5,yy-0.85),13.0*sc,1.5,facecolor=col))
        ax.text(96.5,yy,f"{sc:.2f}",fontsize=6.2,ha="right",va="center",color="#333")
    fig.savefig(FIG+"fig_tool.pdf"); plt.close(fig)

if __name__=="__main__":
    architecture(); deployment(); print("fig1+fig6 ok")
