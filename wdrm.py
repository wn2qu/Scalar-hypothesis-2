"""WDRM 2.0: exploratory causal, sensorimotor, and metacognitive proxies.
NumPy required; SciPy cKDTree optional. See README for assumptions/corrections.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

__version__ = "2.0.0"


def scalar(x, name, minimum=0., maximum=None, strict=False):
    if np.ndim(x) or np.iscomplexobj(x) or isinstance(x, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real scalar.")
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real scalar.") from exc
    if not np.isfinite(v) or v < minimum or (strict and v == minimum):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} {minimum}.")
    if maximum is not None and v > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return v


def integer(x, name, minimum=1):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or x < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(x)


def matrix(x, name, min_rows=2):
    if np.iscomplexobj(x):
        raise ValueError(f"{name} must be real.")
    try:
        a = np.array(x, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] < min_rows or a.shape[1] == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite sample-by-feature data with >= {min_rows} rows.")
    return a


def aligned(*arrays):
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Sample counts must match.")


def protocol(description, randomized):
    if not isinstance(description, str) or not description.strip() or randomized is not True:
        raise ValueError("Explicitly attest independent randomized interventions and describe the protocol.")


def ranks(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts-1)/2)[inverse]


def spearman(x, y):
    for a in (x, y):
        if np.ptp(a) <= 64*np.finfo(float).eps*max(np.max(np.abs(a)), np.finfo(float).tiny):
            return 0.
    return float(np.clip(np.corrcoef(ranks(x), ranks(y))[0, 1], -1, 1))


def entropy(a):
    """Discrete empirical joint entropy; no dense state-space allocation."""
    if not a.shape[1]:
        return 0.
    _, counts = np.unique(a, axis=0, return_counts=True)
    p = counts/counts.sum()
    return -float(p @ np.log(p))


def copula(a):
    cols = [(ranks(a[:, j])-.5)/len(a) for j in range(a.shape[1]) if np.ptp(a[:, j]) > 0]
    return np.column_stack(cols) if cols else np.empty((len(a), 0))


def distances(a, b):
    out = np.zeros((len(a), len(b)))
    for j in range(a.shape[1]):
        np.maximum(out, np.abs(a[:, j, None]-b[None, :, j]), out=out)
    return out


@dataclass(frozen=True)
class WDRMParameters:
    w1: float = .50
    w2: float = .25
    w3: float = .25
    kappa: float = 1e-3
    epsilon_tau: float = 1e-12
    sigma_meta: float = .15
    eta_snr: float = 3.
    knn_k: int = 5
    n_surrogates: int = 19
    surrogate_z: float = 2.
    min_shift: int = 20
    theiler_window: int = 0
    max_exact_units: int = 8
    seed: int = 42

    def __post_init__(self):
        for name in ("w1", "w2", "w3"):
            object.__setattr__(self, name, scalar(getattr(self, name), name, maximum=1, strict=True))
        if not np.isclose(self.w1+self.w2+self.w3, 1, atol=1e-12, rtol=0):
            raise ValueError("Strictly positive weights must sum to one.")
        for name in ("kappa", "epsilon_tau", "sigma_meta", "eta_snr"):
            object.__setattr__(self, name, scalar(getattr(self, name), name, strict=True))
        object.__setattr__(self, "surrogate_z", scalar(self.surrogate_z, "surrogate_z"))
        for name, floor in (("knn_k",1), ("n_surrogates",2), ("min_shift",1),
                            ("theiler_window",0), ("max_exact_units",2), ("seed",0)):
            object.__setattr__(self, name, integer(getattr(self, name), name, floor))
        if self.min_shift <= self.theiler_window:
            raise ValueError("min_shift must exceed theiler_window.")


class InformationEstimator:
    """KSG-1 MI / Frenzel-Pompe CMI on rank-copula data, or discrete CMI.
    Circular shifts are heuristic null controls, not exact conditional tests.
    """
    def __init__(self, params=None, backend="auto"):
        self.params = params or WDRMParameters()
        if backend not in ("auto", "numpy", "scipy"):
            raise ValueError("Unknown k-NN backend.")
        if backend == "scipy" and cKDTree is None:
            raise ImportError("Install scipy for the scipy backend.")
        self.backend = "scipy" if backend != "numpy" and cKDTree is not None else "numpy"

    def _knn(self, x, y, z):
        n, k, w = len(x), self.params.knn_k, self.params.theiler_window
        if n-1-2*w < k:
            raise ValueError("Too few eligible neighbors for k and theiler_window.")
        if not x.shape[1] or not y.shape[1]:
            return 0.
        joint, xz, yz = np.column_stack((x,y,z)), np.column_stack((x,z)), np.column_stack((y,z))
        # Integer digamma values without Euler's constant (it cancels).
        psi = np.zeros(n+1)
        psi[2:] = np.cumsum(1./np.arange(1,n))
        if self.backend == "scipy":
            d, indices = cKDTree(joint).query(joint, k=k+2*w+1, p=np.inf)
            d[np.abs(indices-np.arange(n)[:,None]) <= w] = np.inf
            radius = np.partition(d, k-1, axis=1)[:,k-1]
            if np.any(radius <= 0) or not np.isfinite(radius).all():
                raise ValueError("Degenerate neighborhoods; use discrete mode for repeated values.")
            radius = np.nextafter(radius, 0)
            def counts(a):
                if not a.shape[1]:
                    return np.array([n-(min(n,i+w+1)-max(0,i-w)) for i in range(n)])
                count = cKDTree(a).query_ball_point(a, radius, p=np.inf, return_length=True)
                for i in range(n):
                    near = a[max(0,i-w):min(n,i+w+1)]
                    count[i] -= np.count_nonzero(np.max(np.abs(near-a[i]),axis=1) <= radius[i])
                return count
            nxz, nyz, nz = counts(xz), counts(yz), counts(z)
            return float(psi[k]+np.mean(psi[nz+1]-psi[nxz+1]-psi[nyz+1]))
        if n > 4000:
            raise ValueError("NumPy k-NN fallback limited to 4000 rows; install scipy or explicitly subsample.")
        terms = []
        for start in range(0,n,128):
            stop = min(n,start+128)
            valid = np.abs(np.arange(start,stop)[:,None]-np.arange(n)[None,:]) > w
            d = distances(joint[start:stop],joint)
            d[~valid] = np.inf
            radius = np.partition(d,k-1,axis=1)[:,k-1]
            if np.any(radius <= 0):
                raise ValueError("Degenerate neighborhoods; use discrete mode for repeated values.")
            radius = np.nextafter(radius,0)
            nxz, nyz, nz = [np.sum((distances(a[start:stop],a) <= radius[:,None]) & valid,axis=1)
                           for a in (xz,yz,z)]
            terms.extend(psi[nz+1]-psi[nxz+1]-psi[nyz+1])
        return float(psi[k]+np.mean(terms))

    def estimate(self, x, y, z=None, *, kind="knn", correct=True):
        if kind not in ("knn","discrete") or not isinstance(correct,bool):
            raise ValueError("Use kind='knn' or 'discrete', and boolean correct.")
        x, y = matrix(x,"x"), matrix(y,"y")
        z = np.empty((len(x),0)) if z is None else matrix(z,"z")
        aligned(x,y,z)
        if kind == "knn":
            x,y,z = copula(x),copula(y),copula(z)
            estimate = self._knn
        else:
            def estimate(a,b,c):
                return (entropy(np.column_stack((a,c)))+entropy(np.column_stack((b,c)))
                        -entropy(c)-entropy(np.column_stack((a,b,c))))
        raw = estimate(x,y,z)
        shifts, null = [], []
        if correct:
            available = np.arange(self.params.min_shift,len(x)-self.params.min_shift+1)
            if len(available) < self.params.n_surrogates:
                raise ValueError("Insufficient distinct circular shifts; change sample count or settings.")
            shifts = np.random.default_rng(self.params.seed).choice(available,self.params.n_surrogates,replace=False)
            null = [estimate(np.roll(x,int(s),axis=0),y,z) for s in shifts]
        mean = float(np.mean(null)) if len(null) else 0.
        std = float(np.std(null,ddof=1)) if len(null) else 0.
        amount = max(0.,raw-mean-self.params.surrogate_z*std) if correct else max(0.,raw)
        return {"raw_nats":float(raw),"null_mean_nats":mean,"null_std_nats":std,
                "corrected_nats":float(amount),"surrogate_corrected":correct,
                "shifts":[int(s) for s in shifts],"kind":kind,
                "backend":self.backend if kind=="knn" else "discrete_counts"}


@dataclass
class InterventionalData:
    current: np.ndarray
    next_state: np.ndarray
    protocol: str
    independently_randomized: bool

    def validated(self):
        protocol(self.protocol,self.independently_randomized)
        x,y = matrix(self.current,"current"),matrix(self.next_state,"next_state")
        if x.shape != y.shape or x.shape[1] < 2:
            raise ValueError("Interventional arrays must match and have >= 2 units.")
        return x,y


class KernelRidgeProbe:
    """RBF kernel ridge with training-only input/output scaling."""
    def __init__(self, regularization=1e-3, gamma=None, max_train_rows=2000):
        self.regularization = scalar(regularization,"regularization",strict=True)
        self.gamma = None if gamma is None else scalar(gamma,"gamma",strict=True)
        self.max_train_rows = integer(max_train_rows,"max_train_rows",2)
        self.fitted = False

    def _kernel(self,x,y):
        d2 = np.maximum(0,(x*x).sum(axis=1)[:,None]+(y*y).sum(axis=1)[None,:]-2*x@y.T)
        return np.exp(-self.gamma_fitted*d2)

    def fit(self,x,y):
        x,y = matrix(x,"probe inputs"),matrix(y,"probe targets")
        aligned(x,y)
        if len(x) > self.max_train_rows:
            raise ValueError("Kernel fit exceeds max_train_rows; explicitly subsample.")
        self.x_mean,self.x_scale = x.mean(axis=0),x.std(axis=0)
        self.y_mean,self.y_scale = y.mean(axis=0),y.std(axis=0)
        self.x_scale[self.x_scale==0] = 1
        self.y_scale[self.y_scale==0] = 1
        self.train = (x-self.x_mean)/self.x_scale
        self.gamma_fitted = self.gamma if self.gamma is not None else 1/x.shape[1]
        kernel = self._kernel(self.train,self.train)
        kernel.flat[::len(kernel)+1] += self.regularization*len(x)
        self.weights = np.linalg.solve(kernel,(y-self.y_mean)/self.y_scale)
        if not np.isfinite(self.weights).all():
            raise ValueError("Nonfinite fit; rescale data.")
        self.fitted = True
        return self

    def predict(self,x):
        if not self.fitted:
            raise ValueError("Fit probe before prediction.")
        x = matrix(x,"probe inputs",min_rows=1)
        if x.shape[1] != len(self.x_mean):
            raise ValueError("Probe feature dimension mismatch.")
        x = (x-self.x_mean)/self.x_scale
        pred = np.vstack([self._kernel(x[i:i+256],self.train)@self.weights for i in range(0,len(x),256)])
        pred = pred*self.y_scale+self.y_mean
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite prediction; rescale data.")
        return pred


@dataclass
class MetaProbe:
    abstraction: KernelRidgeProbe
    error_predictor: KernelRidgeProbe
    target_dimension: int
    code_dimension: int


class WDRMFunctional:
    def __init__(self, params=None, *, knn_backend="auto"):
        if params is not None and not isinstance(params,WDRMParameters):
            raise ValueError("params must be WDRMParameters.")
        self.params = params or WDRMParameters()
        self.information = InformationEstimator(self.params,knn_backend)

    def evaluate(self,i_int,r_ground,h_meta):
        values = [scalar(v,n,maximum=1) for v,n in zip((i_int,r_ground,h_meta),("i_int","r_ground","h_meta"))]
        score = 0. if 0 in values else float(np.exp(np.dot((self.params.w1,self.params.w2,self.params.w3),np.log(values))))
        score = float(np.clip(score,0,1))
        return score,dict(zip(("I_int","R_ground","H_meta","Psi_C"),[*values,score]))

    @staticmethod
    def estimate_intrinsic_timescale(states,sample_interval,*,max_lag=None):
        """First interpolated 1/e crossing of mean normalized channel ACF.
        Requires ordered, unperturbed observations, not randomized do-trials.
        """
        a = matrix(states,"ordered states",min_rows=4)
        dt = scalar(sample_interval,"sample_interval",strict=True)
        a /= np.maximum(np.max(np.abs(a),axis=0),np.finfo(float).tiny)
        a -= a.mean(axis=0)
        energy = np.sum(a*a,axis=0)
        a,energy = a[:,energy>0],energy[energy>0]
        if not len(energy):
            raise ValueError("Constant series has no autocorrelation timescale.")
        limit = len(a)//2 if max_lag is None else integer(max_lag,"max_lag")
        if limit >= len(a):
            raise ValueError("max_lag must be below sample count.")
        fft = np.fft.rfft(a,n=2*len(a),axis=0)
        acf = (np.fft.irfft(fft*fft.conj(),n=2*len(a),axis=0)[:limit+1]/energy).mean(axis=1)
        crossing = np.flatnonzero(acf[1:] <= np.exp(-1))
        if not len(crossing):
            raise ValueError("No 1/e crossing observed; collect longer data or report censoring.")
        lag = int(crossing[0]+1)
        fraction = (acf[lag-1]-np.exp(-1))/(acf[lag-1]-acf[lag])
        return float((lag-1+fraction)*dt),{"crossing_lag":lag,"acf_at_crossing":float(acf[lag]),
                                         "active_channels":len(energy),"sample_interval":dt}

    def latency_response(self,phi_norm,latency_tau,tau_auto,*,include_penalty=True):
        phi = scalar(phi_norm,"phi_norm",maximum=1)
        tau = scalar(latency_tau,"latency_tau")
        intrinsic = scalar(tau_auto,"tau_auto",strict=True)
        if not isinstance(include_penalty,bool):
            raise ValueError("include_penalty must be boolean.")
        with np.errstate(over="ignore",under="ignore"):
            ratio = np.float64(intrinsic)/(np.float64(tau)+self.params.epsilon_tau)
            response = np.tanh(ratio)*(np.exp(-self.params.kappa*ratio) if include_penalty else 1)
        return float(phi*response)

    @staticmethod
    def _partitions(m,candidates):
        if candidates is None:
            return [(0,*[i for i in range(1,m) if mask & (1<<(i-1))])
                    for mask in range((1<<(m-1))-1)]
        cuts = set()
        for cut in candidates:
            indices = list(cut)
            if not indices or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,Integral) or i<0 or i>=m for i in indices):
                raise ValueError("Invalid candidate cut indices.")
            if len(set(indices)) != len(indices) or len(indices)==m:
                raise ValueError("Cuts need distinct indices and two nonempty sides.")
            a = set(indices)
            if 0 not in a:
                a = set(range(m))-a
            cuts.add(tuple(sorted(a)))
        if not cuts:
            raise ValueError("At least one candidate cut is required.")
        return sorted(cuts)

    def _transfer(self,source,target,condition,kind,correct):
        diag = self.information.estimate(source,target,condition,kind=kind,correct=correct)
        amount = diag["corrected_nats"]
        if kind=="discrete":
            h = max(0.,entropy(np.column_stack((target,condition)))-entropy(condition))
            value = 0. if h<=1e-12 else float(np.clip(amount/h,0,1))
            diag.update(conditional_entropy_nats=h,normalization="discrete_conditional_entropy")
        else:
            d = max(1,int(np.count_nonzero(np.ptp(target,axis=0)>0)))
            value = float(-np.expm1(-2*amount/d))
            diag.update(target_dimensions=d,normalization="exp_map_per_target_dimension")
        diag["normalized"] = value
        return value,diag

    def compute_causal_integration(self,data,*,latency_tau,tau_auto,kind="knn",correct=True,
                                   candidate_partitions=None,include_penalty=True):
        if not isinstance(data,InterventionalData):
            raise ValueError("Supply InterventionalData with protocol.")
        x,y = data.validated()
        m = x.shape[1]
        if candidate_partitions is None and m>self.params.max_exact_units:
            raise ValueError("Exact MIP exceeds max_exact_units; provide candidate_partitions explicitly.")
        cuts = self._partitions(m,candidate_partitions)
        results = []
        for a in cuts:
            b = tuple(i for i in range(m) if i not in a)
            f,fd = self._transfer(x[:,a],y[:,b],x[:,b],kind,correct)
            r,rd = self._transfer(x[:,b],y[:,a],x[:,a],kind,correct)
            results.append({"A":list(a),"B":list(b),"cut_score":(f+r)/2,"A_to_B":fd,"B_to_A":rd})
        weakest = min(results,key=lambda row:row["cut_score"])
        phi = weakest["cut_score"]
        score = self.latency_response(phi,latency_tau,tau_auto,include_penalty=include_penalty)
        total = (1<<(m-1))-1
        return score,{"Phi_norm":phi,"I_int":score,"MIP":{"A":weakest["A"],"B":weakest["B"]},
                      "search":"exact" if len(cuts)==total else "restricted_upper_bound",
                      "cuts_evaluated":len(cuts),"total_possible_cuts":total,"tau_auto":float(tau_auto),
                      "protocol":data.protocol,"provenance":"caller_attested_not_verified","cuts":results}

    def compute_grounding(self,sensory_now,environment_future,actions,sensory_future,*,
                          action_protocol=None,actions_randomized=False,mode="external",kind="knn",correct=True):
        if mode not in ("external","internal"):
            raise ValueError("mode must be external or internal.")
        s,e,future = matrix(sensory_now,"sensory_now"),matrix(environment_future,"environment_future"),matrix(sensory_future,"sensory_future")
        aligned(s,e,future)
        if actions is None:
            return 0.,{"R_ground":0.,"reason":"no_actions","mode":mode}
        action = matrix(actions,"actions")
        aligned(s,action)
        if np.all(np.ptp(action,axis=0)==0):
            return 0.,{"R_ground":0.,"reason":"constant_actions","mode":mode}
        protocol(action_protocol,actions_randomized)
        p,pd = self._transfer(s,e,action,kind,correct)
        c,cd = self._transfer(action,future,s,kind,correct)
        score = p*c
        return score,{"R_ground":score,"predictive_relevance":pd,"loop_closure":cd,"mode":mode,
                      "protocol":action_protocol,"provenance":"caller_attested_not_verified"}

    @staticmethod
    def fit_meta_probe(target_train,code_train,target_calibration,code_calibration,*,
                       regularization=1e-3,gamma=None,max_train_rows=2000):
        target,code = matrix(target_train,"target_train"),matrix(code_train,"code_train")
        cal,ccode = matrix(target_calibration,"target_calibration"),matrix(code_calibration,"code_calibration")
        aligned(target,code)
        aligned(cal,ccode)
        if target.shape[1]!=cal.shape[1] or code.shape[1]!=ccode.shape[1]:
            raise ValueError("Train/calibration dimensions must match.")
        if code.shape[1]>=target.shape[1]:
            raise ValueError("Compression gate requires 0 < d2 < d1.")
        g = KernelRidgeProbe(regularization,gamma,max_train_rows).fit(code,target)
        errors = np.sum((cal-g.predict(ccode))**2,axis=1)
        h = KernelRidgeProbe(regularization,gamma,max_train_rows).fit(ccode,errors)
        return MetaProbe(g,h,target.shape[1],code.shape[1])

    def compute_metacognition(self,target_test,code_test,probe,*,noise_variance):
        noise = scalar(noise_variance,"noise_variance",strict=True)
        if not isinstance(probe,MetaProbe):
            raise ValueError("Use a fitted MetaProbe.")
        target,code = matrix(target_test,"target_test"),matrix(code_test,"code_test")
        aligned(target,code)
        if target.shape[1]!=probe.target_dimension or code.shape[1]!=probe.code_dimension:
            raise ValueError("Test dimensions differ from fitted probe.")
        variance = float(np.mean((target-target.mean(axis=0))**2))
        floor = self.params.eta_snr*noise
        if not np.isfinite(variance) or not np.isfinite(floor):
            raise ValueError("Variance overflow; rescale targets and noise_variance together.")
        if code.shape[1]>=target.shape[1] or variance<=floor:
            return 0.,{"H_meta":0.,"F_abs":0.,"F_cal":0.,"reason":"compression_or_snr_gate","variance":variance}
        error = target-probe.abstraction.predict(code)
        errors = np.sum(error*error,axis=1)
        nmse = float(np.mean(error*error)/variance)
        if not np.isfinite(nmse) or not np.isfinite(errors).all():
            raise ValueError("Error overflow; rescale targets and noise_variance together.")
        r2 = max(0.,1-nmse)
        with np.errstate(over="ignore",under="ignore"):
            fidelity = float(r2*np.exp(-.5*np.square(np.sqrt(nmse)/self.params.sigma_meta)))
        predicted = np.maximum(0.,probe.error_predictor.predict(code).ravel())
        rho = spearman(predicted,errors)
        calibration = max(0.,rho)
        score = fidelity*calibration
        return score,{"H_meta":score,"F_abs":fidelity,"F_cal":calibration,"spearman_rho":rho,
                      "NMSE":nmse,"R2_gated":r2,"variance":variance,"snr":variance/noise,
                      "calibration_kind":"rank_error_sensitivity_not_absolute_calibration"}

    def latency_phase_sweep(self,phi_norm,tau_auto,r_ground,h_meta,tau_exponents,*,include_penalty=True):
        exponents = np.asarray(tau_exponents)
        if exponents.ndim!=1 or not len(exponents):
            raise ValueError("tau_exponents must be a nonempty vector.")
        rows = []
        for exponent in exponents:
            if not np.isfinite(exponent):
                raise ValueError("Latency exponents must be finite.")
            with np.errstate(over="ignore",under="ignore"):
                tau = float(np.power(10.,exponent))
            if not np.isfinite(tau) or tau<=0:
                raise ValueError("Exponent must yield positive finite latency.")
            integration = self.latency_response(phi_norm,tau,tau_auto,include_penalty=include_penalty)
            score,_ = self.evaluate(integration,r_ground,h_meta)
            rows.append({"log10_tau":float(exponent),"tau_sec":tau,"I_int":integration,"Psi_C":score})
        return rows


def demo():
    import json
    rng = np.random.default_rng(42)
    model = WDRMFunctional(WDRMParameters(n_surrogates=7,min_shift=10,sigma_meta=.5))
    x = np.tile(np.array([[0,0],[0,1],[1,0],[1,1]]),(80,1))
    rng.shuffle(x)
    data = InterventionalData(x,x[:,::-1],"Synthetic independent fair-bit resets; next units swap.",True)
    trace = np.zeros((2000,2))
    for t in range(1,len(trace)):
        trace[t] = .8*trace[t-1]+rng.normal(size=2)
    tau_auto,_ = model.estimate_intrinsic_timescale(trace,.01)
    integration,idiag = model.compute_causal_integration(data,latency_tau=tau_auto/2,tau_auto=tau_auto,
                                                        kind="discrete",correct=False)
    s,a = rng.normal(size=(2,400))
    e = s+.3*a+rng.normal(scale=.3,size=400)
    sf = .5*s+a+rng.normal(scale=.3,size=400)
    grounding,_ = model.compute_grounding(s,e,a,sf,action_protocol="Synthetic independently randomized actions.",
                                          actions_randomized=True)
    z,q = rng.uniform(-2,2,900),rng.uniform(-1,1,900)
    code = np.column_stack((z,q))
    target = np.column_stack((np.sin(z),np.cos(z),z,z*z,np.sin(2*z),np.cos(2*z)))
    target += rng.normal(size=target.shape)*(.02+.10*(q[:,None]+1)/2)
    probe = model.fit_meta_probe(target[:300],code[:300],target[300:600],code[300:600],regularization=1e-5)
    meta,mdiag = model.compute_metacognition(target[600:],code[600:],probe,noise_variance=1e-4)
    _,details = model.evaluate(integration,grounding,meta)
    details.update(Phi_norm=idiag["Phi_norm"],tau_auto=tau_auto,F_abs=mdiag["F_abs"],
                   F_cal=mdiag["F_cal"],backend=model.information.backend)
    print("WDRM 2.0 synthetic component demonstrations; not one measured conscious system.")
    print(json.dumps(details,indent=2))
    print("Missing-pillar controls:",[model.evaluate(*v)[0] for v in ((0,1,1),(1,0,1),(1,1,0))])


if __name__ == "__main__":
    demo()

