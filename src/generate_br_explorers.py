#!/usr/bin/env python3
"""
Genera L{p}_{q}.html para todos los casos en BR_polynomials.json.
También genera br_index.html con enlaces a todos los archivos.
"""
from pathlib import Path
import json, numpy as np

SCRIPT_DIR = Path(__file__).parent
DATA_PATH  = SCRIPT_DIR.parent / 'data' / 'BR_polynomials.json'

# ── Helpers geométricos ───────────────────────────────────────────────────────
def build_tile(p, s1, s2):
    tile, tile_inv = {}, {}
    seen = set()
    for dist in range(400):
        for x in range(-dist, dist+1):
            y_abs = dist - abs(x)
            for y in ([y_abs, -y_abs] if y_abs > 0 else [0]):
                k = (x*s1 + y*s2) % p
                if k not in seen:
                    seen.add(k); tile[(x,y)] = k; tile_inv[k] = (x,y)
        if len(seen) == p: break
    return tile, tile_inv

def boundary_polys(tile):
    cs = set(tile.keys())
    segs = []
    for (x, y) in cs:
        if (x, y - 1) not in cs:
            segs.append(((x, y), (x + 1, y)))
        if (x + 1, y) not in cs:
            segs.append(((x + 1, y), (x + 1, y + 1)))
        if (x, y + 1) not in cs:
            segs.append(((x + 1, y + 1), (x, y + 1)))
        if (x - 1, y) not in cs:
            segs.append(((x, y + 1), (x, y)))

    nexts = {}
    for a, b in segs:
        nexts.setdefault(a, []).append(b)

    def pick_next(cur, prev):
        cand = [v for v in nexts.get(cur, []) if v != prev]
        if not cand:
            return None
        # Keep traversal deterministic on ambiguous corners.
        cand.sort()
        return cand[0]

    polys = []
    used = set()
    for a, b in sorted(segs):
        if (a, b) in used:
            continue
        loop = [a]
        st = a
        prev, cur = a, b
        used.add((a, b))
        guard = 0
        while guard < 100000:
            guard += 1
            loop.append(cur)
            if cur == st:
                break
            nxt = pick_next(cur, prev)
            if nxt is None:
                break
            used.add((cur, nxt))
            prev, cur = cur, nxt
        if len(loop) >= 4 and loop[-1] == loop[0]:
            polys.append(loop)

    return polys

def tor(phi, theta, R=2.6, r=1.0):
    x = (R + r*np.cos(phi))*np.cos(theta)
    y = (R + r*np.cos(phi))*np.sin(theta)
    z = r*np.sin(phi); return x, y, z

def torus_path(lp, p, q, n=60):
    xs, ys, zs = [], [], []
    for i in range(len(lp)-1):
        a0,b0 = lp[i]; a1,b1 = lp[i+1]
        ts = np.linspace(0,1,n,endpoint=False)
        phi   = 2*np.pi*(a0+(a1-a0)*ts + q*(b0+(b1-b0)*ts))/p
        theta = 2*np.pi*(b0+(b1-b0)*ts)
        px,py,pz = tor(phi, theta)
        xs += px.tolist(); ys += py.tolist(); zs += pz.tolist()
    a,b = lp[-1]
    phi0   = 2*np.pi*(a + q*b)/p
    theta0 = 2*np.pi*b
    px,py,pz = tor(phi0, theta0)
    xs.append(float(px)); ys.append(float(py)); zs.append(float(pz))
    return xs, ys, zs

def hsl(h, s=75, l=42): return f'hsl({h},{s}%,{l}%)'

def path_edges(x0, y0, x1, y1, s1, s2, p):
    """Rectilinear path (horiz first), returns list of edge IDs."""
    edges = []
    dx = 1 if x1 >= x0 else -1
    x = x0
    while x != x1:
        k = (min(x, x+dx)*s1 + y0*s2) % p
        edges.append(f'e-{k}'); x += dx
    dy = 1 if y1 >= y0 else -1
    y = y0
    while y != y1:
        k = (x1*s1 + min(y, y+dy)*s2) % p
        edges.append(f'f-{k}'); y += dy
    return edges

def compute_steiner(p, s1, s2):
    """Steiner tree in the null lattice of G(p; ±s1, ±s2)."""
    bound = p + 2
    null_vecs = sorted(
        (abs(x)+abs(y), x, y)
        for y in range(-bound, bound+1)
        for x in range(-bound, bound+1)
        if (x,y) != (0,0) and (x*s1+y*s2) % p == 0
    )
    if not null_vecs: return []
    _, ux, uy = null_vecs[0]
    vx, vy = p, 0
    for _, cx, cy in null_vecs[1:]:
        if ux*cy - uy*cx != 0: vx, vy = cx, cy; break
    sx = sorted([0, ux, vx])[1]
    sy = sorted([0, uy, vy])[1]
    edges = []
    if (sx, sy) != (0, 0): edges += path_edges(0,0, sx,sy, s1,s2, p)
    edges += path_edges(sx,sy, ux,uy, s1,s2, p)
    edges += path_edges(sx,sy, vx,vy, s1,s2, p)
    return list(dict.fromkeys(edges))

# ── Construir todos los datos para un caso ────────────────────────────────────
def build_case(p, q, br_case):
    s1, s2 = 1, q

    tile, tile_inv = build_tile(p, s1, s2)
    DOMAIN_POLYS = boundary_polys(tile)
    TILE_INV_JS  = {k: list(v) for k,v in tile_inv.items()}

    # Toro
    u_s,v_s = np.linspace(0,2*np.pi,50), np.linspace(0,2*np.pi,50)
    uu,vv   = np.meshgrid(u_s,v_s)
    Tx,Ty,Tz = tor(uu, vv)
    TORUS_SURF = {'x':Tx.tolist(),'y':Ty.tolist(),'z':Tz.tolist()}
    vi = np.arange(p, dtype=float)
    pv = 2*np.pi*(vi)/p;  tv = np.zeros(p)   # vertex i at (a=i, b=0)
    Vx,Vy,Vz = tor(pv, tv)
    VERT_POS = {'x':Vx.tolist(),'y':Vy.tolist(),'z':Vz.tolist(),
                'text':[f'v{i}' for i in range(p)]}

    # Aristas en el toro: e = paso ±1, f = paso ±q
    EDGE_PATHS = {}
    for i in range(p):
        xi,yi = tile_inv[i]
        EDGE_PATHS[f'e-{i}'] = dict(zip('xyz', torus_path([(xi,yi),(xi+1,yi)], p, q)))
        EDGE_PATHS[f'f-{i}'] = dict(zip('xyz', torus_path([(xi,yi),(xi,yi+1)], p, q)))

    EDGE_COLORS = {f'f-{i}': hsl(int(360*i/p))       for i in range(p)}
    EDGE_COLORS.update({f'e-{i}': hsl(int(360*i/p),55,32) for i in range(p)})

    # Perfil por z: para cada (t,z), máximo y mínimo exponente en x y coef. en el máximo
    terms = br_case.get('terms', [])
    display_terms = [(s, t, z, c) for s, t, z, c in terms if z >= 0]
    TERM_COEFF = {f'{s}|{t}|{z}': c for s,t,z,c in display_terms}
    ZT_PROFILE = {}
    TZ_PROFILE = {}
    ST_PROFILE = {}
    Z_SLICE_TERMS = {}
    for s, t, z, c in display_terms:
        zkey, tkey = str(z), str(t)
        zbucket = ZT_PROFILE.setdefault(zkey, {})
        rec = zbucket.get(tkey)
        if rec is None:
            zbucket[tkey] = {'sMax': s, 'sMin': s, 'coefAtMax': c}
        else:
            if s > rec['sMax']:
                rec['sMax'] = s
                rec['coefAtMax'] = c
            if s < rec['sMin']:
                rec['sMin'] = s

        t_bucket = TZ_PROFILE.setdefault(tkey, {})
        t_rec = t_bucket.get(zkey)
        if t_rec is None:
            t_bucket[zkey] = {'sMax': s, 'sMin': s, 'coefAtMax': c}
        else:
            if s > t_rec['sMax']:
                t_rec['sMax'] = s
                t_rec['coefAtMax'] = c
            if s < t_rec['sMin']:
                t_rec['sMin'] = s

        st_key = f'{s}|{t}'
        ST_PROFILE.setdefault(st_key, []).append([z, c])
        Z_SLICE_TERMS.setdefault(str(z), []).append([s, t, c])

    for st_key, zcs in ST_PROFILE.items():
        zcs.sort(key=lambda r: r[0])
    for z_key, recs in Z_SLICE_TERMS.items():
        recs.sort(key=lambda r: (r[1], r[0]))

    # Steiner
    STEINER_EDGES = compute_steiner(p, s1, s2)

    # Dimensiones de la cuadrícula (centrada en el origen y ~150% más amplia)
    half_span = max(3, int(np.ceil(0.75 * p)))
    XN, XX = -half_span, half_span
    YN, YX = -half_span, half_span

    CELL  = max(28, min(56, 600//p))
    MGl   = max(42, CELL + 2)
    MGt, MGr, MGb = 14, 14, 48

    # Diamante: caja simétrica respecto al origen para una vista más balanceada
    coords  = list(tile_inv.values())
    max_abs_cx = max(abs(c[0]) for c in coords)
    max_abs_cy = max(abs(c[1]) for c in coords)
    CELLD   = max(18, min(40, 480//max(p,8)))
    D_XN, D_XX = -max_abs_cx-1, max_abs_cx+1
    D_YN, D_YX = -max_abs_cy-1, max_abs_cy+1
    D_NX = D_XX - D_XN + 1
    D_NY = D_YX - D_YN + 1
    D_OX = -D_XN
    D_OY = D_YX

    grid_h  = (YX-YN+1)*CELL + MGt + MGb

    DATA_JS = f"""
const P={p}, Q={q}, S1={s1}, S2={s2};
const XN={XN}, XX={XX}, YN={YN}, YX={YX};
const CELL={CELL}, MGl={MGl}, MGt={MGt}, MGr={MGr}, MGb={MGb};
const CELLD={CELLD}, D_NX={D_NX}, D_NY={D_NY}, D_OX={D_OX}, D_OY={D_OY};
const DOMAIN_POLYS={json.dumps(DOMAIN_POLYS)};
const TILE_INV={json.dumps(TILE_INV_JS)};
const EDGE_PATHS={json.dumps(EDGE_PATHS)};
const EDGE_COLORS={json.dumps(EDGE_COLORS)};
const TORUS_SURF={json.dumps(TORUS_SURF)};
const VERT_POS={json.dumps(VERT_POS)};
const STEINER_EDGES={json.dumps(STEINER_EDGES)};
const TERM_COEFF={json.dumps(TERM_COEFF)};
const ZT_PROFILE={json.dumps(ZT_PROFILE)};
const TZ_PROFILE={json.dumps(TZ_PROFILE)};
const ST_PROFILE={json.dumps(ST_PROFILE)};
const Z_SLICE_TERMS={json.dumps(Z_SLICE_TERMS)};
"""
    return DATA_JS, grid_h

# ── Plantilla JavaScript (genérica, lee constantes de DATA_JS) ────────────────
JS = r"""
function alpha(d){
    const i=Math.floor(d/4),j=d%4;
    if(j===0) return 4*((i+1)%P)+2;
    if(j===1) return 4*((i+Q)%P)+3;
    if(j===2) return 4*((i-1+P)%P)+0;
               return 4*((i-Q+P)%P)+1;
}
function sigma(d){ return 4*Math.floor(d/4)+(d%4+1)%4; }
function edgeToDarts(eid){
    const[type,idx]=eid.split('-'); const i=parseInt(idx);
    return type==='e'?[4*i,4*((i+1)%P)+2]:[4*i+1,4*((i+Q)%P)+3];
}
function computeBR(sel){
    const inF=new Set();
    for(const eid of sel) for(const d of edgeToDarts(eid)) inF.add(d);
    const vis=new Uint8Array(4*P); let fF=0;
    for(let st=0;st<4*P;st++){
        if(!vis[st]){ fF++; let c=st;
            while(!vis[c]){ vis[c]=1; c=sigma(inF.has(c)?alpha(c):c); }
        }
    }
    const par=Array.from({length:P},(_,i)=>i);
    function find(x){while(par[x]!==x){par[x]=par[par[x]];x=par[x];}return x;}
    for(const eid of sel){
        const[type,idx]=eid.split('-'); const i=parseInt(idx);
        const j=type==='e'?(i+1)%P:(i+Q)%P;
        const ri=find(i),rj=find(j); if(ri!==rj) par[ri]=rj;
    }
    const kF=new Set(Array.from({length:P},(_,i)=>find(i))).size;
    const eF=sel.size, nF=eF-P+kF;
    const rG=P-1, rF=P-kF, s=rG-rF, t=nF;
    const zExp=kF-fF+nF, gF=zExp/2;
    return{s,t,zExp,gF,kF,nF,eF,fF,rG,rF};
}
function termCoeff(s,t,z){ return TERM_COEFF[`${s}|${t}|${z}`] ?? 0; }
function coeffsForST(s,t){ return ST_PROFILE[`${s}|${t}`] ?? []; }
function fmtNorm(raw){
    const norm=raw/P;
    return Number.isInteger(norm) ? String(norm) : norm.toFixed(3);
}
function fmtCellCoeff(raw){
    const a=Math.abs(raw);
    if(a>=1e5){
        const exp=Math.floor(Math.log10(a));
        const man=(raw/Math.pow(10,exp)).toFixed(2);
        return `${man}e${exp}`;
    }
    return String(raw);
}

const gCanvas=document.getElementById('grid');
function gxy(gx,gy){ return [MGl+(gx-XN)*CELL, MGt+(YX-gy)*CELL]; }
function canvasToGrid(px,py){ return [(px-MGl)/CELL+XN, YX-(py-MGt)/CELL]; }

function drawGrid(sel){
    const NX=XX-XN+1, NY=YX-YN+1;
    gCanvas.width=MGl+NX*CELL+MGr; gCanvas.height=MGt+NY*CELL+MGb;
    const gc=gCanvas.getContext('2d');
    gc.clearRect(0,0,gCanvas.width,gCanvas.height);
    gc.fillStyle='#ffffff'; gc.fillRect(MGl,MGt,NX*CELL,NY*CELL);
    gc.strokeStyle='#d0d0d0'; gc.lineWidth=0.6;
    for(let ix=0;ix<=NX;ix++){
        const[px,py]=gxy(XN+ix,YN);
        gc.beginPath(); gc.moveTo(px,py); gc.lineTo(px,py-NY*CELL); gc.stroke();
    }
    for(let iy=0;iy<=NY;iy++){
        const[px,py]=gxy(XN,YN+iy);
        gc.beginPath(); gc.moveTo(px,py); gc.lineTo(px+NX*CELL,py); gc.stroke();
    }
    gc.fillStyle='#333'; gc.font=`${CELL*0.40}px sans-serif`;
    for(let x=XN;x<=XX;x++){
        const[px,]=gxy(x,YN);
        gc.textAlign='center'; gc.textBaseline='top';
        gc.fillText(x,px,MGt+(YX-YN)*CELL+7);
    }
    for(let y=YN;y<=YX;y++){
        const[,py]=gxy(XN,y);
        gc.textAlign='right'; gc.textBaseline='middle';
        gc.fillText(y,MGl-6,py);
    }
    gc.fillStyle='#111'; gc.font=`bold italic ${CELL*0.36}px serif`;
    gc.textAlign='center'; gc.textBaseline='top';
    gc.fillText(`x  (arista e, \u00b11)`,MGl+NX*CELL/2, MGt+NY*CELL+26);
    gc.save(); gc.translate(8,MGt+NY*CELL/2);
    gc.rotate(-Math.PI/2); gc.textAlign='center'; gc.textBaseline='top';
    gc.fillText(`y  (arista f, \u00b1${Q})`,0,0); gc.restore();

    // Steiner tree hint
    gc.lineCap='round'; gc.strokeStyle='#bbb'; gc.lineWidth=3.5;
    gc.setLineDash([7,5]);
    for(const eid of STEINER_EDGES){
        if(sel.has(eid)) continue;
        const[type,idx]=eid.split('-'); const ei=parseInt(idx);
        for(let y=YN;y<=YX;y++) for(let x=XN;x<=XX;x++){
            const k=((x*S1+y*S2)%P+P)%P;
            if(type==='e' && k===ei && x<XX){
                const[px1,py]=gxy(x,y);const[px2,]=gxy(x+1,y);
                gc.beginPath();gc.moveTo(px1,py);gc.lineTo(px2,py);gc.stroke();
            }
            if(type==='f' && k===ei && y<YX){
                const[px,py1]=gxy(x,y);const[,py2]=gxy(x,y+1);
                gc.beginPath();gc.moveTo(px,py1);gc.lineTo(px,py2);gc.stroke();
            }
        }
    }
    gc.setLineDash([]);
    // Aristas seleccionadas
    gc.lineCap='round';
    for(const eid of sel){
        gc.strokeStyle=EDGE_COLORS[eid]; gc.lineWidth=5.5;
        const[type,idx]=eid.split('-'); const ei=parseInt(idx);
        for(let y=YN;y<=YX;y++) for(let x=XN;x<=XX;x++){
            const k=((x*S1+y*S2)%P+P)%P;
            if(type==='e' && k===ei && x<XX){
                const[px1,py]=gxy(x,y);const[px2,]=gxy(x+1,y);
                gc.beginPath();gc.moveTo(px1,py);gc.lineTo(px2,py);gc.stroke();
            }
            if(type==='f' && k===ei && y<YX){
                const[px,py1]=gxy(x,y);const[,py2]=gxy(x,y+1);
                gc.beginPath();gc.moveTo(px,py1);gc.lineTo(px,py2);gc.stroke();
            }
        }
    }
}

const dCanvas=document.getElementById('diamond');
function drawDiamond(sel){
    const active=new Set();
    for(const eid of sel){
        const[type,idx]=eid.split('-'); const i=parseInt(idx);
        active.add(i); active.add(type==='e'?(i+1)%P:(i+Q)%P);
    }
    dCanvas.width=D_NX*CELLD+20; dCanvas.height=D_NY*CELLD+68;
    const dc=dCanvas.getContext('2d');
    dc.clearRect(0,0,dCanvas.width,dCanvas.height);
    dc.fillStyle='#1d3557'; dc.font=`bold 14px serif`;
    dc.textAlign='center'; dc.textBaseline='top';
    dc.fillText('Dominio diamante / borde tipo L',dCanvas.width/2,2);
    dc.strokeStyle='#e3e7ee'; dc.lineWidth=0.8;
    for(let gx=0;gx<=D_NX;gx++){
        const x=10+gx*CELLD;
        dc.beginPath(); dc.moveTo(x,18); dc.lineTo(x,18+D_NY*CELLD); dc.stroke();
    }
    for(let gy=0;gy<=D_NY;gy++){
        const y=18+gy*CELLD;
        dc.beginPath(); dc.moveTo(10,y); dc.lineTo(10+D_NX*CELLD,y); dc.stroke();
    }
    dc.strokeStyle='#1d3557'; dc.lineWidth=2.3;
    for(const loop of DOMAIN_POLYS){
        const poly=loop.map(([x,y])=>[(x+D_OX)*CELLD+10,(D_OY-y)*CELLD+18]);
        if(poly.length<2) continue;
        dc.beginPath();
        for(let i=0;i<poly.length;i++){
            const [px,py]=poly[i];
            if(i===0) dc.moveTo(px,py);
            else dc.lineTo(px,py);
        }
        dc.stroke();
    }
    for(const[ks,pos] of Object.entries(TILE_INV)){
        const k=parseInt(ks); const[cx,cy]=pos;
        const px=(cx+D_OX)*CELLD+10, py=(D_OY-cy)*CELLD+18;
        const isActive=active.has(k), isNull=k===0;
        dc.fillStyle=isNull&&isActive?'#8b1a1a':isNull?'#e63946':isActive?'#457b9d':'#e8eef4';
        dc.fillRect(px,py,CELLD,CELLD);
        dc.strokeStyle=isActive?'#1d1d1d':'#aaaaaa'; dc.lineWidth=isActive?2:0.7;
        dc.strokeRect(px,py,CELLD,CELLD);
        dc.fillStyle=isActive?'white':isNull?'white':'#555';
        dc.font=`${isActive?'bold ':' '}${CELLD*0.42}px serif`;
        dc.textAlign='center'; dc.textBaseline='middle';
        dc.fillText(k,px+CELLD/2,py+CELLD/2);
    }
    const [ox,oy]=[(D_OX)*CELLD+10+CELLD/2,(D_OY)*CELLD+18+CELLD/2];
    dc.fillStyle='#111';
    dc.beginPath(); dc.arc(ox,oy,2.8,0,2*Math.PI); dc.fill();
    dc.font='13px sans-serif'; dc.textBaseline='middle';
    dc.fillStyle='#457b9d'; dc.fillRect(10,dCanvas.height-43,12,12);
    dc.fillStyle='#444'; dc.fillText('v\u00e9rtice activo',25,dCanvas.height-37);
    dc.fillStyle='#e8eef4'; dc.strokeStyle='#aaa'; dc.lineWidth=0.7;
    dc.fillRect(120,dCanvas.height-43,12,12); dc.strokeRect(120,dCanvas.height-43,12,12);
    dc.fillStyle='#444'; dc.fillText('aislado',135,dCanvas.height-37);
    const activeList=[...active].sort((a,b)=>a-b).join(', ') || '-';
    dc.fillStyle='#444'; dc.font='12px monospace';
    dc.fillText(`consumidos: ${activeList}`,10,dCanvas.height-17);
}

let torusReady=false;
function initTorus(){
    const surf={type:'surface',x:TORUS_SURF.x,y:TORUS_SURF.y,z:TORUS_SURF.z,
        opacity:0.16,colorscale:'Blues',showscale:false,hoverinfo:'skip'};
    const verts={type:'scatter3d',mode:'markers+text',
        x:VERT_POS.x,y:VERT_POS.y,z:VERT_POS.z,text:VERT_POS.text,
        textposition:'top center',textfont:{size:9,color:'#4c566a'},
        marker:{size:4,color:'#4c566a'},hoverinfo:'text',showlegend:false};
    const edges={type:'scatter3d',mode:'lines',x:[],y:[],z:[],
        line:{color:'#e63946',width:7},showlegend:false};
    const layout={margin:{l:0,r:0,t:32,b:0},
        scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},
               aspectmode:'data',bgcolor:'#f8faff',
               camera:{eye:{x:1.6,y:1.4,z:1.1}}},
        paper_bgcolor:'white',
        title:{text:'Grafo en el toro T\u00b2',font:{size:13},x:0.5}};
    Plotly.newPlot('torus-div',[surf,verts,edges],layout,
                   {responsive:true,displayModeBar:false})
          .then(()=>{torusReady=true;});
}
function updateTorus(sel){
    if(!torusReady) return;
    const X=[],Y=[],Z=[];
    for(const eid of sel){
        const ep=EDGE_PATHS[eid];
        X.push(...ep.x,null);Y.push(...ep.y,null);Z.push(...ep.z,null);
    }
    Plotly.restyle('torus-div',{x:[X],y:[Y],z:[Z]},[2]);
}

function updateExponentRibbon(br){
    document.getElementById('br-ribbon').innerHTML=
    `<div class="rtext">`+
    `Un <b>subgrafo abarcador F \u2286 G</b> conserva todos los v\u00e9rtices y elige cualquier subconjunto de aristas. `+
    `Sus invariantes son: <b>k(F)</b> componentes conexas, <b>e(F)</b> aristas, <b>v(F)</b> v\u00e9rtices, <b>f(F)</b> caras del ribbon y <b>g(F)</b> g\u00e9nero. `+
    `De ellos se derivan <b>r(F) = v(F) \u2212 k(F)</b> (rango) y <b>n(F) = e(F) \u2212 r(F)</b> (nulidad), `+
    `que determinan los exponentes del monomio de BR correspondiente.`+
    `</div>`+
    `<div class="rvals">`+
    `<span class="rbadge">n(F)=<b>${br.nF}</b></span>`+
    `<span class="rbadge">s=r(G)-r(F)=<b>${br.s}</b></span>`+
    `<span class="rbadge">exp(z)=<b>${br.zExp}</b></span>`+
    `<span class="rbadge">g(F)=<b>${br.gF}</b></span>`+
    `</div>`;
}

function updateBRDisplay(sel){
    const br=computeBR(sel);
    const coefRaw=br.eF===0 ? 1 : termCoeff(br.s,br.t,br.zExp);
    const coefNorm=fmtNorm(coefRaw);
    const repMono=br.eF===0 ? '1' : `X<sup>${br.s}</sup>&thinsp;y<sup>${br.t}</sup>&thinsp;z<sup>${br.zExp}</sup>`;
    updateExponentRibbon(br);
    const tags=[...sel].map(e=>`<span class="etag" style="background:${EDGE_COLORS[e]}">${e}</span>`).join(' ')||'<em style="color:#888">ninguna</em>';
    document.getElementById('br-display').innerHTML=`
      <table class="brtable">
        <tr><th colspan="2">Subgrafo seleccionado</th></tr>
                <tr><td>Representante</td><td>${repMono}</td></tr>
        <tr><td>Aristas</td><td>${tags}</td></tr>
        <tr><td>e(F)</td><td>${br.eF} / ${2*P}</td></tr>
        <tr><td>k(F)</td><td>${br.kF}</td></tr>
        <tr><td>f(F)</td><td>${br.fF}</td></tr>
        <tr><td>g(F)</td><td>${br.gF}</td></tr>
        <tr><td>Coef.</td><td><b>${coefRaw}</b></td></tr>
        <tr><td>Coef/p</td><td><b>${coefNorm}</b></td></tr>
      </table>`;
}

function updateCrestDisplay(sel){
    const br=computeBR(sel);
    const zKey=String(br.zExp);
    const sliceTerms=Z_SLICE_TERMS[zKey] || [];
    if(sliceTerms.length===0){
        document.getElementById('crest-display').innerHTML=
            `<table class="cresttable">`+
            `<tr><th colspan="4">Rebanada z<sup>${br.zExp}</sup></th></tr>`+
            `<tr><td colspan="4">No hay t\u00e9rminos en la base de datos con este exponente de z.</td></tr>`+
            `</table>`;
        return;
    }
    const sVals=sliceTerms.map(r=>r[0]), tVals=sliceTerms.map(r=>r[1]);
    const sMin=Math.min(...sVals), sMax=Math.max(...sVals);
    const tMin=Math.min(...tVals), tMax=Math.max(...tVals);
    const nS=sMax-sMin+1, nT=tMax-tMin+1;
    const maxDigits=Math.max(...sliceTerms.map(r=>fmtCellCoeff(r[2]).length),1);
    const cellH=Math.max(18, Math.min(34, Math.floor(460/Math.max(nS,8))));
    const idealW=Math.ceil(maxDigits*9.2)+18;
    const fitW=Math.floor(1400/Math.max(nT,8));
    const cellW=Math.max(46, Math.min(220, Math.min(idealW, fitW)));
    const cellFont=Math.max(13, Math.min(Math.floor(cellH*0.66), Math.floor((cellW-10)/Math.max(maxDigits*0.56,1))));
    const ml=50, mt=24, mr=20, mb=46;
    const width=ml+nT*cellW+mr, height=mt+nS*cellH+mb;

    document.getElementById('crest-display').innerHTML=
        `<div class="slicebox">`+
        `<div class="slicetitle">Rebanada z<sup>${br.zExp}</sup> : punto (a,b) = y<sup>a</sup>X<sup>b</sup></div>`+
        `<canvas id="zslice-canvas" width="${width}" height="${height}"></canvas>`+
        `<div class="slicelegend">`+
        `<span><b>Verde</b>: representante X<sup>${br.s}</sup>y<sup>${br.t}</sup>z<sup>${br.zExp}</sup></span> &nbsp;|&nbsp;`+
        `<span><b>Ámbar</b>: cascada de la expansión en x</span> &nbsp;|&nbsp;`+
        `<span><b>Azul</b>: término presente fuera de la cascada</span> &nbsp;|&nbsp;`+
        `<span>${tMin} \u2264 t \u2264 ${tMax}, &nbsp; ${sMin} \u2264 s \u2264 ${sMax}</span>`+
        `</div></div>`;

    const cv=document.getElementById('zslice-canvas');
    const cx=cv.getContext('2d');
    const coeffByST = new Map(sliceTerms.map(([s,t,c])=>[`${s}|${t}`, c]));
    cx.clearRect(0,0,width,height);
    cx.fillStyle='#ffffff';
    cx.fillRect(ml,mt,nT*cellW,nS*cellH);
    cx.strokeStyle='#d6dee8';
    cx.lineWidth=0.8;
    for(let i=0;i<=nT;i++){
        const x=ml+i*cellW;
        cx.beginPath(); cx.moveTo(x,mt); cx.lineTo(x,mt+nS*cellH); cx.stroke();
    }
    for(let j=0;j<=nS;j++){
        const y=mt+j*cellH;
        cx.beginPath(); cx.moveTo(ml,y); cx.lineTo(ml+nT*cellW,y); cx.stroke();
    }

    for(const [s,t,c] of sliceTerms){
        const ix=t-tMin;
        const iy=sMax-s;
        const px=ml+ix*cellW, py=mt+iy*cellH;
        const isRep=(s===br.s && t===br.t);
        const inCascade=(t===br.t && s>=0 && s<=br.s);
        cx.fillStyle=isRep ? '#2a9d4b' : inCascade ? '#c9892f' : '#4f84b8';
        cx.fillRect(px+2,py+2,cellW-4,cellH-4);
        cx.strokeStyle=isRep ? '#155724' : inCascade ? '#7a4f1c' : '#355c7d';
        cx.lineWidth=isRep ? 2.4 : inCascade ? 1.8 : 1.0;
        cx.strokeRect(px+2,py+2,cellW-4,cellH-4);
        cx.fillStyle='#ffffff';
        cx.font=`bold ${cellFont}px sans-serif`;
        cx.textAlign='center';
        cx.textBaseline='middle';
        const lbl=fmtCellCoeff(c);
        cx.fillText(lbl, px + cellW/2, py + cellH/2);
    }


    const tickStepT = nT > 18 ? 2 : 1;
    const tickStepS = nS > 18 ? 2 : 1;
    cx.fillStyle='#374151';
    cx.font='14px sans-serif';
    cx.textAlign='center';
    cx.textBaseline='top';
    for(let t=tMin;t<=tMax;t+=tickStepT){
        const x=ml+(t-tMin+0.5)*cellW;
        cx.fillText(String(t),x,mt+nS*cellH+6);
    }
    cx.textAlign='right';
    cx.textBaseline='middle';
    for(let s=sMin;s<=sMax;s+=tickStepS){
        const y=mt+(sMax-s+0.5)*cellH;
        cx.fillText(String(s),ml-6,y);
    }

    cx.fillStyle='#1f2937';
    cx.font='bold 14px sans-serif';
    cx.textAlign='center';
    cx.textBaseline='top';
    cx.fillText('a = exponente de y (nulidad t)', ml+nT*cellW/2, height-18);
    cx.save();
    cx.translate(14, mt+nS*cellH/2);
    cx.rotate(-Math.PI/2);
    cx.textAlign='center';
    cx.textBaseline='top';
    cx.fillText('b = exponente de X (conectividad s)',0,0);
    cx.restore();
}

const sel=new Set();
function nearestEdge(gx,gy){
    const fracx=gx-Math.floor(gx), fracy=gy-Math.floor(gy);
    const dv=Math.min(fracx,1-fracx), dh=Math.min(fracy,1-fracy);
    const TH=0.30;
    if(dh<dv && dh<TH){
        const nodeX=Math.floor(gx), nodeY=Math.round(gy);
        return `e-${((nodeX*S1+nodeY*S2)%P+P)%P}`;
    }
    if(dv<dh && dv<TH){
        const nodeX=Math.round(gx), nodeY=Math.floor(gy);
        return `f-${((nodeX*S1+nodeY*S2)%P+P)%P}`;
    }
    return null;
}
gCanvas.addEventListener('mousedown',function(e){
    e.preventDefault();
    const r=gCanvas.getBoundingClientRect();
    const[gx,gy]=canvasToGrid(e.clientX-r.left,e.clientY-r.top);
    const eid=nearestEdge(gx,gy); if(!eid) return;
    if(e.button===0) sel.add(eid); else if(e.button===2) sel.delete(eid);
    drawGrid(sel);updateTorus(sel);updateBRDisplay(sel);updateCrestDisplay(sel);
});
gCanvas.addEventListener('contextmenu',e=>e.preventDefault());
document.getElementById('btn-clear').addEventListener('click',()=>{
    sel.clear();drawGrid(sel);updateTorus(sel);updateBRDisplay(sel);updateCrestDisplay(sel);
});
document.getElementById('btn-complete').addEventListener('click',()=>{
    sel.clear();
    for(let i=0;i<P;i++){ sel.add(`e-${i}`); sel.add(`f-${i}`); }
    drawGrid(sel);updateTorus(sel);updateBRDisplay(sel);updateCrestDisplay(sel);
});
window.addEventListener('load',()=>{
    drawGrid(sel);initTorus();updateBRDisplay(sel);updateCrestDisplay(sel);
});
"""

# ── Plantilla HTML ────────────────────────────────────────────────────────────
def make_html(p, q, DATA_JS, grid_h):
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Explorador BR \u2014 G({p}; \u00b11, \u00b1{q})</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
*{{box-sizing:border-box;}}
body{{margin:0;font-family:'Times New Roman',serif;background:#fafafa;font-size:19px;}}
#header{{background:#1d3557;color:white;padding:9px 18px;font-size:1.22em;
         display:flex;align-items:center;gap:20px;flex-wrap:wrap;}}
#header span{{font-size:0.9em;color:#a8c8e8;}}
#btn-clear{{background:#e63946;color:white;border:none;padding:4px 12px;
                border-radius:4px;cursor:pointer;font-size:0.96em;font-family:inherit;}}
#btn-clear:hover{{background:#c0262f;}}
#btn-complete{{background:#2a9d4b;color:white;border:none;padding:4px 12px;
                    border-radius:4px;cursor:pointer;font-size:0.96em;font-family:inherit;}}
#btn-complete:hover{{background:#1f7d3b;}}
#top{{display:flex;gap:0;align-items:flex-start;}}
#br-ribbon{{margin:6px 8px 0 8px;padding:8px 10px;border:1px solid #ccd9ea;border-radius:4px;
       background:linear-gradient(180deg,#f7fbff 0%,#eef5ff 100%);}}
.rtext{{color:#243b57;font-size:1.03em;line-height:1.38;}}
.rvals{{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;}}
.rbadge{{display:inline-block;background:#ffffff;border:1px solid #c9d7ea;color:#1f3857;
    padding:4px 9px;border-radius:999px;font-size:0.96em;white-space:nowrap;}}
#left{{flex:0 0 auto;padding:8px;}}
#right{{flex:1;padding:8px;}}
#torus-div{{width:100%;height:{grid_h}px;border:1px solid #ccc;border-radius:4px;}}
#grid{{display:block;cursor:crosshair;border:1px solid #ccc;border-radius:4px;}}
#bottom{{display:block;gap:12px;padding:10px 8px;align-items:flex-start;
         border-top:2px solid #1d3557;margin-top:4px;}}
#br-section{{display:flex;gap:12px;align-items:flex-start;}}
#br-display{{flex:0 0 470px;margin-bottom:0;}}
#crest-display{{flex:1;min-width:0;margin-top:0;overflow-x:auto;}}
.brtable{{border-collapse:collapse;width:100%;display:table;min-width:430px;max-width:none;font-size:1.12em;background:#fff;border:1px solid #d5dfeb;border-radius:4px;overflow:hidden;}}
.brtable th{{text-align:left;padding:6px 10px;background:#eef4fb;
             border-bottom:2px solid #1d3557;font-size:1.2em;}}
.brtable td{{padding:5px 10px;border-bottom:1px solid #dde;}}
.brtable tr:hover td{{background:#f5f9ff;}}
.etag{{display:inline-block;color:white;padding:1px 5px;border-radius:3px;
    font-size:0.8em;font-family:monospace;margin:1px;}}
.cresttable{{border-collapse:collapse;width:100%;font-size:1.1em;}}
.cresttable th{{text-align:left;padding:5px 10px;background:#dde7f4;
               border-bottom:2px solid #1d3557;font-family:'Times New Roman',serif;}}
.cresttable td{{padding:4px 10px;border-bottom:1px solid #eee;font-family:monospace;}}
.cresttable tr:hover td{{background:#f5f9ff;}}
.crest-ok{{background:#d4edda!important;}}
.crest-cur{{background:#fff3cd!important;}}
.slicebox{{border:1px solid #d7e0eb;background:#fcfdff;border-radius:4px;padding:8px;}}
.slicetitle{{font-size:1.14em;color:#1d3557;margin-bottom:6px;}}
.slicelegend{{font-size:1.01em;color:#445;line-height:1.38;margin-top:6px;}}
</style>
</head>
<body>
<div id="header">
  <b>G({p};&nbsp;&plusmn;1,&nbsp;&plusmn;{q}) &mdash; Explorador BR</b>
  <span>
        <b>Click izq:</b> anadir arista &nbsp;
        <b>Click der:</b> quitar arista &nbsp;
        <button id="btn-clear">Limpiar</button>
        <button id="btn-complete">Completar G</button>
  </span>
  <span style="margin-left:auto">arista e = paso &plusmn;1 &nbsp;|&nbsp; arista f = paso &plusmn;{q}</span>
</div>
<div id="top">
  <div id="left"><canvas id="grid"></canvas></div>
  <div id="right"><div id="torus-div"></div></div>
</div>
<div id="br-ribbon"></div>
<div id="bottom">
  <div id="br-section"><div id="br-display"></div><div id="crest-display"></div></div>
</div>
<script>
{DATA_JS}
{JS}
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cases = json.loads(DATA_PATH.read_text())['cases']
    generated = []

    for case in cases:
        p, q = case['p'], case['q']
        print(f'  L({p},{q})...', end=' ', flush=True)
        DATA_JS, grid_h = build_case(p, q, case)
        html = make_html(p, q, DATA_JS, grid_h)
        fname = f'L{p}_{q}.html'
        out = SCRIPT_DIR / fname
        out.write_text(html, encoding='utf-8')
        kb = out.stat().st_size // 1024
        print(f'{fname}  ({kb} KB)')
        generated.append((p, q, fname, kb))

    # Índice
    rows = '\n'.join(
        f'  <tr><td><a href="{fn}">L({p},{q})</a></td>'
        f'<td>G({p}; &plusmn;1, &plusmn;{q})</td>'
        f'<td style="text-align:right">{kb}&thinsp;KB</td></tr>'
        for p, q, fn, kb in generated
    )
    index = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Exploradores BR &mdash; Grafos en Espacios Lente</title>
<style>
body{{font-family:'Times New Roman',serif;max-width:640px;margin:40px auto;background:#fafafa;line-height:1.5;}}
h1{{color:#1d3557;font-size:1.5em;margin-bottom:0.3em;}}
p{{color:#444;font-size:0.95em;margin-top:0.2em;}}
table{{border-collapse:collapse;width:100%;margin-top:1.2em;}}
th{{background:#1d3557;color:white;padding:6px 14px;text-align:left;}}
td{{padding:5px 14px;border-bottom:1px solid #dde;}}
a{{color:#1d3557;text-decoration:none;}}
a:hover{{text-decoration:underline;}}
tr:hover td{{background:#eef4fb;}}
</style>
</head>
<body>
<h1>Exploradores BR &mdash; Espacios Lente</h1>
<p>Polinomio de Boll&oacute;b&aacute;s&ndash;Riordan del grafo de Heegaard G(p;&nbsp;&plusmn;1,&nbsp;&plusmn;q)
   en el toro T&sup2;.</p>
<table>
<tr><th>Espacio</th><th>Grafo</th><th>Tama&ntilde;o</th></tr>
{rows}
</table>
</body>
</html>"""
    idx_out = SCRIPT_DIR / 'br_index.html'
    idx_out.write_text(index, encoding='utf-8')
    print(f'\nÍndice: {idx_out}')
    print(f'Total: {len(generated)} exploradores.')

if __name__ == '__main__':
    main()
