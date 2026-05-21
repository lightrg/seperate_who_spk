
import os ,sys ,json ,math ,logging ,contextlib ,tempfile
from pathlib import Path
from typing import Optional ,List ,Dict ,Tuple
from itertools import permutations as _perms
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
import torch .nn as nn
import torch .nn .functional as F
import torchaudio
from torch .optim import AdamW

try :
    if hasattr (torchaudio ,"set_audio_backend"):
        for _backend in ("sox_io","soundfile"):
            try :
                torchaudio .set_audio_backend (_backend )
                logging .getLogger ("ft").info (f"Using torchaudio backend: {_backend }")
                break
            except Exception :
                continue
except Exception :
    pass

def _load_audio (path :str ):
    try :
        return torchaudio .load (path )
    except Exception as e :
        message =str (e )
        if "torchcodec"in message or "TorchCodec"in message :
            try :
                if hasattr (torchaudio ,"set_audio_backend"):
                    for _backend in ("sox_io","soundfile"):
                        try :
                            torchaudio .set_audio_backend (_backend )
                            return torchaudio .load (path )
                        except Exception :
                            continue
            except Exception :
                pass

            try :
                import soundfile as sf
                data ,sr =sf .read (path ,dtype ="float32")
                data =np .asarray (data )
                if data .ndim ==1 :
                    data =data [np .newaxis ,:]
                else :
                    data =data .T
                return torch .from_numpy (data ),sr
            except Exception :
                pass

            try :
                from scipy .io import wavfile
                sr ,data =wavfile .read (path )
                if data .ndim ==1 :
                    data =data [np .newaxis ,:]
                else :
                    data =data .T
                if data .dtype !=np .float32 :
                    if np .issubdtype (data .dtype ,np .integer ):
                        data =data .astype (np .float32 )/np .iinfo (data .dtype ).max
                    else :
                        data =data .astype (np .float32 )
                return torch .from_numpy (data ),sr
            except Exception :
                pass

        raise
from pyannote .audio .utils .powerset import Powerset
from scipy .optimize import linear_sum_assignment

from ft_config import (
FinetuneConfig ,PhaseConfig ,build_phase_configs ,
TEST_MODE ,SOURCE_SELF ,SOURCE_TIER_A ,
SELF_LABELED_DIR ,SELF_LABELED_META ,
DATA_MIX_DIR ,DATA_MIX_META ,
VAL_DATA_DIR ,TEST_DATA_DIR ,VAL_LABELED_DIR ,TEST_LABELED_DIR ,
)
from ft_dataloader import (
Sample ,
scan_self_labeled ,scan_all_sources ,scan_fixed_split_dir ,
train_val_split ,parse_rttm ,
build_train_loader ,build_val_loader ,
AudioAugmenter ,SR ,FRAME_SHIFT_SEC ,
)

def _get_device_type (device :torch .device )->str :
    if device .type =="cuda":return "cuda"
    if device .type =="mps":return "mps"
    return "cpu"

def _resolve_amp_dtype (device_type :str ,amp_dtype :Optional [str ]):
    if device_type !="cuda":
        return None
    name =str (amp_dtype or "float16").lower ()
    if name in {"bf16","bfloat16"}:
        return torch .bfloat16
    return torch .float16

def _use_grad_scaler (enabled :bool ,device_type :str ,amp_dtype :Optional [str ])->bool :
    return bool (enabled and device_type =="cuda"and _resolve_amp_dtype (device_type ,amp_dtype )==torch .float16 )

try :
    from torch .amp import GradScaler as _GradScaler ,autocast as _autocast_fn
    def _make_scaler (enabled ,device_type ,amp_dtype =None ):
        return _GradScaler (device_type ,enabled =_use_grad_scaler (enabled ,device_type ,amp_dtype ))
    def _autocast (enabled ,device_type ="cpu",amp_dtype =None ):
        kwargs ={"enabled":enabled }
        dtype =_resolve_amp_dtype (device_type ,amp_dtype )
        if dtype is not None :
            kwargs ["dtype"]=dtype
        return _autocast_fn (device_type ,**kwargs )
except ImportError :
    from torch .cuda .amp import GradScaler as _GradScaler ,autocast as _autocast_fn
    def _make_scaler (enabled ,device_type ,amp_dtype =None ):
        return _GradScaler (enabled =_use_grad_scaler (enabled ,device_type ,amp_dtype ))
    def _autocast (enabled ,device_type ="cpu",amp_dtype =None ):
        if device_type !="cuda":
            return contextlib .nullcontext ()
        dtype =_resolve_amp_dtype (device_type ,amp_dtype )
        return _autocast_fn (enabled =enabled ,dtype =dtype )

_log_dir =Path (__file__ ).resolve ().parents [2 ]/"results"/"logs"
_log_dir .mkdir (parents =True ,exist_ok =True )
logging .basicConfig (
level =logging .INFO ,
format ="%(asctime)s [%(levelname)s] %(message)s",
handlers =[
logging .StreamHandler (),
logging .FileHandler (_log_dir /"finetune_v2.log",encoding ="utf-8"),
],
)
log =logging .getLogger ("ft")

HEAD_PARAM_NAMES ={
"_seg_model.proj.weight",
"_seg_model.proj.bias",
"_seg_model.lnorm.weight",
"_seg_model.lnorm.bias",
"_seg_model.classifier.weight",
"_seg_model.classifier.bias",
}

def _is_head_param (name :str )->bool :
    return name in HEAD_PARAM_NAMES

def load_ecapa_model ():

    try :
        try :
            from speechbrain .inference import EncoderClassifier
        except ImportError :
            from speechbrain .pretrained import EncoderClassifier

        _ecapa_savedir =str (Path (tempfile .gettempdir ())/"sb_ecapa")
        ecapa =EncoderClassifier .from_hparams (
        source ="speechbrain/spkrec-ecapa-voxceleb",
        savedir =_ecapa_savedir ,
        )
        ecapa .eval ()
        log .info ("  [enroll] ECAPA-TDNN loaded ")
        return ecapa
    except Exception as e :
        log .warning (f"  [enroll] ECAPA not available ({e }). EnrollConf metric disabled.")
        return None

def load_enrollment_centroids (enrollment_dir :str ,ecapa ,device )->Dict [str ,np .ndarray ]:

    centroids :Dict [str ,np .ndarray ]={}
    enroll =Path (enrollment_dir )
    if not enroll .exists ()or ecapa is None :
        return centroids
    for spk_dir in sorted (enroll .iterdir ()):
        if not spk_dir .is_dir ():continue
        spk_id =spk_dir .name
        combined =spk_dir /f"{spk_id }.wav"
        wav_file =combined if combined .exists ()else next (spk_dir .glob ("*.wav"),None )
        if wav_file is None :continue
        try :
            wav ,sr =_load_audio (str (wav_file ))
            if wav .shape [0 ]>1 :wav =wav .mean (0 ,keepdim =True )
            if sr !=16000 :wav =torchaudio .functional .resample (wav ,sr ,16000 )
            if wav .shape [1 ]<3200 :continue
            with torch .no_grad ():
                emb =ecapa .encode_batch (wav .to (device )).squeeze ().cpu ().numpy ().astype (np .float32 )
            centroids [spk_id ]=emb /(np .linalg .norm (emb )+1e-8 )
        except :pass
    return centroids

def compute_enroll_confusion (
logits :torch .Tensor ,
labels :torch .Tensor ,
n_frames :int ,
centroids :Dict [str ,np .ndarray ],
wav_path :str ,
ecapa ,
device :torch .device ,
threshold :float =0.5 ,
)->Optional [float ]:

    if ecapa is None or not centroids or n_frames <=0 :
        return None
    centroid_list =list (centroids .items ())
    centroid_mat =np .stack ([e for _ ,e in centroid_list ])

    probs =decode_probs (logits [:n_frames ]).cpu ().numpy ()
    pred =(probs >threshold ).astype (np .float32 )
    gt =labels [:n_frames ].cpu ().numpy ()
    n_act =pred .sum (axis =1 )

    try :
        wav ,sr =_load_audio (wav_path )
        if wav .shape [0 ]>1 :wav =wav .mean (0 ,keepdim =True )
        if sr !=16000 :wav =torchaudio .functional .resample (wav ,sr ,16000 )
    except :return None

    conf_frames =tot_frames =0
    i =0
    while i <n_frames :
        if n_act [i ]!=1 :i +=1 ;continue
        j =i
        while j <n_frames and n_act [j ]==1 :j +=1
        run =j -i
        if run >=20 :
            s_smp ,e_smp =int (i *FRAME_SHIFT_SEC *16000 ),int (j *FRAME_SHIFT_SEC *16000 )
            if e_smp -s_smp >=3200 and e_smp <=wav .shape [1 ]:
                chunk =wav [:,s_smp :e_smp ].to (device )
                try :
                    with torch .no_grad ():
                        emb =ecapa .encode_batch (chunk ).squeeze ().cpu ().numpy ().astype (np .float32 )
                    emb =emb /(np .linalg .norm (emb )+1e-8 )
                    best_idx =int (np .argmax (centroid_mat @emb ))
                    pred_ch =int (np .argmax (pred [i :j ].mean (axis =0 )))
                    gt_run =gt [i :j ]
                    if gt_run .sum ()>0 :
                        gt_ch =int (np .argmax (gt_run .mean (axis =0 )))
                        if pred_ch !=gt_ch and best_idx !=gt_ch :
                            conf_frames +=run
                    tot_frames +=run
                except :pass
        i =j
    return conf_frames /tot_frames *100.0 if tot_frames >0 else None

class DiariZenWrapper (nn .Module ):
    def __init__ (self ,pipeline ,max_speakers :int =4 ):
        super ().__init__ ()
        self .pipeline =pipeline
        self .max_speakers =max_speakers
        self ._seg_model =None
        self ._wavlm_layers :Optional [nn .ModuleList ]=None
        self ._logits_dim =max_speakers
        self ._batched_forward_warned =False
        self ._probe ()

    def _probe (self ):
        candidate =getattr (self .pipeline ,"model",None )
        if candidate is None :
            raise RuntimeError ("Cannot find pipeline.model")
        inner =getattr (candidate ,"model",candidate )
        self ._seg_model =inner
        log .info (f"  [probe] seg_model: {type (inner ).__name__ }")

        if hasattr (inner ,"classifier")and isinstance (inner .classifier ,nn .Linear ):
            old_head =inner .classifier
            in_features =old_head .in_features
            out_features =old_head .out_features
            log .info (f"  [probe] Original classifier: Linear({in_features }, {out_features })")
            if out_features !=self .max_speakers :
                new_head =nn .Linear (in_features ,self .max_speakers )
                _init_multilabel_head_from_powerset (old_head ,new_head ,self .max_speakers )
                inner .classifier =new_head
                log .info (f"  [probe] Replaced classifier → multilabel Linear({in_features }, {self .max_speakers })")
            inner .activation =nn .Identity ()
            self ._logits_dim =self .max_speakers
        else :
            raise RuntimeError ("Cannot find nn.Linear classifier on segmentation model.")

        for attr in ["wavlm","wavlm_model","encoder","wav2vec","feature_extractor","backbone"]:
            m =getattr (inner ,attr ,None )
            if m and isinstance (m ,nn .Module ):
                for sub in ["encoder","transformer","backbone"]:
                    s =getattr (m ,sub ,m )
                    if s is m and sub !="encoder":
                        continue
                    for sub2 in ["transformer","encoder",""]:
                        s2 =getattr (s ,sub2 ,s )if sub2 else s
                        for la in ["layers","transformer_layers","blocks"]:
                            layers =getattr (s2 ,la ,None )
                            if isinstance (layers ,nn .ModuleList ):
                                self ._wavlm_layers =layers
                                log .info (f"  [probe] WavLM layers found: {len (layers )} via {attr }.{sub }.{sub2 }.{la }")
                                break
                        if self ._wavlm_layers :
                            break
                    if self ._wavlm_layers :
                        break
                break

    def wavlm_layers (self ):
        return self ._wavlm_layers

    def _extract_logits (self ,raw_out )->torch .Tensor :
        if isinstance (raw_out ,tuple ):
            logits =raw_out [0 ]
        elif isinstance (raw_out ,dict ):
            logits =raw_out .get ("logits",next (iter (raw_out .values ())))
        else :
            logits =raw_out

        if logits .ndim ==3 :
            return logits
        if logits .ndim ==2 :
            return logits .unsqueeze (0 )
        if logits .ndim <2 :
            return logits .view (1 ,-1 ,1 )
        return logits

    def _forward_one (self ,wav :torch .Tensor )->torch .Tensor :
        wav_3d =wav .view (1 ,1 ,-1 )
        try :
            with torch .set_grad_enabled (torch .is_grad_enabled ()):
                raw_out =self ._seg_model (wav_3d )
        except Exception as e :
            raise RuntimeError (f"Model forward failed: {e }")
        return self ._extract_logits (raw_out ).squeeze (0 )

    def forward (self ,waveform :torch .Tensor )->torch .Tensor :
        if waveform .ndim ==2 :
            waveform =waveform .unsqueeze (1 )
        try :
            with torch .set_grad_enabled (torch .is_grad_enabled ()):
                raw_out =self ._seg_model (waveform .contiguous ())
            return self ._extract_logits (raw_out )
        except Exception as e :
            B =waveform .shape [0 ]
            if B ==1 :
                raise RuntimeError (f"Model forward failed: {e }")
            if not self ._batched_forward_warned :
                log .warning (f"  [forward] Batched forward failed ({e }). Falling back to per-sample forward.")
                self ._batched_forward_warned =True
            outs =[self ._forward_one (waveform [b ])for b in range (B )]
            max_T =max (x .shape [0 ]for x in outs )
            logits =waveform .new_zeros (B ,max_T ,self ._logits_dim )
            for b ,x in enumerate (outs ):
                logits [b ,:x .shape [0 ],:]=x
            return logits

_PERM_CACHE :Dict [int ,List ]={}
def get_perms (n :int )->List :
    if n not in _PERM_CACHE :_PERM_CACHE [n ]=list (_perms (range (n )))
    return _PERM_CACHE [n ]

def _init_multilabel_head_from_powerset (old_head :nn .Linear ,new_head :nn .Linear ,max_speakers :int )->None :

    if old_head .out_features ==new_head .out_features :
        with torch .no_grad ():
            new_head .weight .copy_ (old_head .weight )
            if old_head .bias is not None and new_head .bias is not None :
                new_head .bias .copy_ (old_head .bias )
        log .info (
        f"  [head_init] Same size ({old_head .out_features }) → direct copy "
        )
        return

    expected_powerset =11
    if old_head .out_features ==expected_powerset and new_head .out_features ==max_speakers :
        log .info (
        f"  [head_init] Powerset-11 → multilabel-{max_speakers }: "
        f"unpacking via Powerset mapping "
        )
        with torch .no_grad ():
            power =Powerset (num_classes =max_speakers ,max_set_size =2 )
            mapping =power .mapping .to (dtype =old_head .weight .dtype ,device =old_head .weight .device )
            silence =(mapping .sum (-1 )==0 )
            for s in range (max_speakers ):
                mask =mapping [:,s ]>0
                if mask .any ():
                    new_head .weight [s ].copy_ (old_head .weight [mask ].mean (0 ))
                    if old_head .bias is not None and new_head .bias is not None :
                        new_head .bias [s ].copy_ (old_head .bias [mask ].mean ())
                else :
                    nn .init .xavier_uniform_ (new_head .weight [s :s +1 ])
                    if new_head .bias is not None :
                        new_head .bias [s ].zero_ ()
            if silence .any ():
                silence_w =old_head .weight [silence ].mean (0 ,keepdim =True )
                new_head .weight .data .sub_ (0.10 *silence_w )
        return

    log .warning (
    f"\n{'!'*70 }\n"
    f"  [head_init] UNEXPECTED HEAD SIZE: "
    f"old={old_head .out_features }, new={new_head .out_features }, max_speakers={max_speakers }\n"
    f"  Pretrained classifier weights CANNOT be reused — falling back to Xavier init.\n"
    f"  WavLM encoder weights are still preserved.\n"
    f"  Expected old_head.out_features == {expected_powerset } for clean powerset transfer.\n"
    f"  Actual: {old_head .out_features }. Check pretrained model config.\n"
    f"{'!'*70 }"
    )
    with torch .no_grad ():
        nn .init .xavier_uniform_ (new_head .weight )
        if new_head .bias is not None :
            nn .init .zeros_ (new_head .bias )

_POWER_CACHE :Dict [tuple ,object ]={}

def decode_probs (logits :torch .Tensor ,n_spk :int =4 )->torch .Tensor :

    if logits .shape [-1 ]==11 and n_spk >=4 :
        key =(logits .device .type ,str (logits .dtype ))
        if key not in _POWER_CACHE :
            _POWER_CACHE [key ]=Powerset (num_classes =4 ,max_set_size =2 )
        power =_POWER_CACHE [key ]
        mapping =power .mapping .to (device =logits .device ,dtype =logits .dtype )
        probs_all =torch .softmax (logits ,dim =-1 )
        return torch .matmul (probs_all ,mapping )
    return torch .sigmoid (logits [...,:n_spk ])

def decode_probs_hard (logits :torch .Tensor ,n_spk :int =4 ,threshold :float =0.5 )->torch .Tensor :

    if logits .shape [-1 ]==11 and n_spk >=4 :
        key =(logits .device .type ,str (logits .dtype ))
        if key not in _POWER_CACHE :
            _POWER_CACHE [key ]=Powerset (num_classes =4 ,max_set_size =2 )
        power =_POWER_CACHE [key ]
        mapping =power .mapping .to (device =logits .device ,dtype =logits .dtype )
        cls_idx =logits .argmax (dim =-1 )
        return mapping [cls_idx ].float ()
    return (torch .sigmoid (logits [...,:n_spk ])>threshold ).float ()

def pit_loss (
logits :torch .Tensor ,
labels :torch .Tensor ,
n_frames :torch .Tensor ,
phase :PhaseConfig ,
)->torch .Tensor :

    B ,_ ,K =labels .shape
    losses =[]

    for b in range (B ):
        L =int (n_frames [b ].item ())
        if L <=0 :
            losses .append (logits [b ].sum ()*0.0 )
            continue

        log_b =logits [b ,:L ,:K ]
        lbl_b =labels [b ,:L ,:K ].to (logits .device ).float ()

        n_act =lbl_b .sum (-1 )
        w =torch .ones (L ,device =logits .device ,dtype =log_b .dtype )
        w [n_act ==0 ]=phase .silence_loss_weight
        w [n_act ==2 ]=phase .overlap_loss_weight
        w [n_act >=3 ]=phase .triple_loss_weight

        ign_frm =int (phase .boundary_ignore_sec /FRAME_SHIFT_SEC )
        if ign_frm >0 :

            w [:min (ign_frm ,L )]*=0.25
            w [max (0 ,L -ign_frm ):]*=0.25

        best_loss =None
        for perm in get_perms (K ):
            lbl_perm =lbl_b [:,list (perm )]
            bce_per_frame =F .binary_cross_entropy_with_logits (
            log_b ,lbl_perm ,reduction ="none"
            ).mean (-1 )
            perm_loss =(bce_per_frame *w ).sum ()/w .sum ().clamp (min =1e-8 )
            if best_loss is None or perm_loss .item ()<best_loss .item ():
                best_loss =perm_loss
        losses .append (best_loss )

    return torch .stack (losses ).mean ()if losses else logits .sum ()*0.0

def pit_der_batch (
logits :torch .Tensor ,
labels :torch .Tensor ,
n_frames :torch .Tensor ,
load_speakers :Optional [torch .Tensor ]=None ,
threshold :float =0.5 ,
)->Dict [str ,float ]:

    S_max =labels .shape [-1 ]

    probs =decode_probs_hard (logits ,n_spk =S_max ,threshold =threshold )

    miss =fa =conf =total =0.0

    for b in range (logits .shape [0 ]):
        L =int (n_frames [b ].item ())
        if L <=0 :
            continue

        n_spk =int (load_speakers [b ].item ())if load_speakers is not None else S_max
        n_spk =max (1 ,min (n_spk ,S_max ))

        p =probs [b ,:L ,:S_max ]
        g =labels [b ,:L ,:S_max ].float ()

        p_bin =p
        cost_matrix =torch .zeros ((S_max ,S_max ))
        for i in range (S_max ):
            gi =g [:,i ]
            for j in range (S_max ):
                cost_matrix [i ,j ]=(gi -p_bin [:,j ]).abs ().sum ().item ()

        from scipy .optimize import linear_sum_assignment
        row_ind ,col_ind =linear_sum_assignment (cost_matrix .numpy ())

        inv_perm =np .zeros (S_max ,dtype =int )
        for i ,j in zip (row_ind ,col_ind ):
            inv_perm [j ]=i
        g_perm =g [:,inv_perm ]

        pred_sum =p_bin .sum (-1 )
        gold_sum =g_perm .sum (-1 )

        total +=(gold_sum >0 ).float ().sum ().item ()

        miss +=F .relu (gold_sum -pred_sum ).sum ().item ()
        fa +=F .relu (pred_sum -gold_sum ).sum ().item ()
        conf +=F .relu (torch .min (pred_sum ,gold_sum )-(p_bin *g_perm ).sum (-1 )).sum ().item ()

    return {
    "miss_frames":miss ,
    "fa_frames":fa ,
    "conf_frames":conf ,
    "total_frames":max (total ,1e-8 ),
    }

def apply_freeze (model :DiariZenWrapper ,strategy :str ,unfreeze_top_n :int ):
    for p in model .parameters ():
        p .requires_grad =False

    head_unlocked =[]
    for name ,p in model .named_parameters ():
        if _is_head_param (name ):
            p .requires_grad =True
            head_unlocked .append (name )
    log .info (f"  [freeze] Head params unlocked: {head_unlocked }")

    if strategy =="none":
        for p in model .parameters ():
            p .requires_grad =True
    elif strategy =="top_N":
        layers =model .wavlm_layers ()
        if layers :
            n =len (layers )
            for i ,layer in enumerate (layers ):
                if i >=max (0 ,n -unfreeze_top_n ):
                    for p in layer .parameters ():
                        p .requires_grad =True
                    log .info (f"  [freeze] Unfrozen WavLM layer {i }/{n -1 }")
        else :
            log .warning ("  [freeze] _wavlm_layers is None! top_N unfreeze skipped. Only head params will be trained.")
    t =sum (p .numel ()for p in model .parameters ()if p .requires_grad )
    n =sum (p .numel ()for p in model .parameters ())
    log .info (f"  [freeze] {strategy } | trainable: {t :,}/{n :,} ({100 *t /max (n ,1 ):.1f}%)")

def build_optimizer (model :DiariZenWrapper ,phase :PhaseConfig )->AdamW :
    head_names =[(nm ,p )for nm ,p in model .named_parameters ()if p .requires_grad and _is_head_param (nm )]
    other_names =[(nm ,p )for nm ,p in model .named_parameters ()if p .requires_grad and not _is_head_param (nm )]

    head_p =[p for _ ,p in head_names ]
    other_p =[p for _ ,p in other_names ]

    log .info (f"  [optimizer] HEAD params ({len (head_p )} tensors):")
    for nm ,_ in head_names :
        log .info (f"    {nm }")
    log .info (f"  [optimizer] OTHER trainable params ({len (other_p )} tensors):")
    for nm ,_ in other_names [:8 ]:
        log .info (f"    {nm }")
    if len (other_names )>8 :
        log .info (f"    ... ({len (other_names )-8 } more)")

    groups =[]
    if head_p :
        groups .append ({"params":head_p ,"lr":phase .lr })
    if other_p :
        groups .append ({"params":other_p ,"lr":phase .lr *0.2 })
    if not groups :
        raise RuntimeError ("No trainable parameters found. Check freeze strategy.")
    log .info (
    f"  [optimizer] groups={len (groups )} | "
    +" | ".join (f"lr={g ['lr']:.2e} ({len (g ['params'])} tensors)"for g in groups )
    )
    return AdamW (groups ,weight_decay =phase .weight_decay ,betas =(0.9 ,0.999 ),eps =1e-8 )

class WarmupCosine :

    def __init__ (self ,opt ,warmup :int ,total :int ,min_r :float =0.05 ):
        self .opt ,self .warmup ,self .total ,self .min_r ,self ._step =opt ,warmup ,total ,min_r ,0
        self ._base =[g ["lr"]for g in opt .param_groups ]

    def step (self ):
        self ._step +=1 ;s =self ._step
        for i ,g in enumerate (self .opt .param_groups ):
            base =self ._base [i ]
            if s <self .warmup :g ["lr"]=base *s /max (self .warmup ,1 )
            else :
                p =(s -self .warmup )/max (self .total -self .warmup ,1 )
                g ["lr"]=base *(self .min_r +(1 -self .min_r )*0.5 *(1 +math .cos (math .pi *p )))

    def get_lr (self ):return [g ["lr"]for g in self .opt .param_groups ]

class ValRegistry :

    def __init__ (self ,val_files :List [Sample ],cfg :FinetuneConfig ):
        from ft_dataloader import build_val_loader
        self .cfg =cfg
        self .loaders :Dict [Tuple [str ,str ],object ]={}

        if val_files :

            self .loaders [("ALL","ALL")]=build_val_loader (val_files ,cfg )

        log .info (f"  [ValRegistry] Simplified mode: {len (val_files )} files total in aggregate loader.")

@torch .no_grad ()
def run_validation (
model :DiariZenWrapper ,
val_registry :ValRegistry ,
cfg :FinetuneConfig ,
phase :PhaseConfig ,
device :torch .device ,
device_type :str ,
ecapa ,
epoch :int ,
model_cpu_ref =None ,
)->Tuple [float ,Dict ]:

    model .eval ()
    all_results :Dict [str ,Dict ]={}
    total_val_loss =0.0
    total_val_der =0.0

    if device_type =="mps":
        if model_cpu_ref is None :

            from diarizen .pipelines .inference import DiariZenPipeline as _DZP
            _pl =_DZP .from_pretrained (cfg .pretrained_model )
            model_cpu_ref =DiariZenWrapper (_pl ,max_speakers =4 ).cpu ().eval ()

        cpu_sd ={k :v .cpu ()for k ,v in model .state_dict ().items ()}
        model_cpu_ref .load_state_dict (cpu_sd )
        model_der =model_cpu_ref
    else :
        model_der =model

    thr_grid =[float (t )for t in getattr (cfg ,"local_der_threshold_grid",[0.50 ])]
    base_thr =float (getattr (cfg ,"local_der_threshold",0.50 ))
    if base_thr not in thr_grid :
        thr_grid .append (base_thr )

    for (src ,spk ),loader in val_registry .loaders .items ():
        split_name =f"{src }/{spk }"
        split_loss =0.0
        n_samples_total =0

        der_acc ={
        float (thr ):{"miss":0.0 ,"fa":0.0 ,"conf":0.0 ,"total":0.0 }
        for thr in thr_grid
        }

        n_val_crops =0
        max_crops =cfg .max_val_crops if hasattr (cfg ,"max_val_crops")else 0
        use_non_blocking =device_type =="cuda"
        amp_dtype =getattr (cfg ,"amp_dtype","float16")
        for batch in tqdm (loader ,desc =f"   | Val ({split_name })",leave =False ,dynamic_ncols =True ):
            waveform =batch ["waveform"].to (device ,non_blocking =use_non_blocking )
            labels =batch ["labels"].to (device ,non_blocking =use_non_blocking )
            n_frames =batch ["n_frames"].to (device ,non_blocking =use_non_blocking )
            load_spk =batch .get ("load_speakers",batch .get ("n_speakers",None ))

            with _autocast (cfg .fp16 and device_type =="cuda",device_type ,amp_dtype ):
                logits =model (waveform )

            if n_val_crops ==0 :
                log .info (f"   [debug] wav: min={waveform .min ():.4f}, max={waveform .max ():.4f}, mean={waveform .mean ():.4f}")
                log .info (f"   [debug] logits: min={logits .min ():.4f}, max={logits .max ():.4f}, mean={logits .mean ():.4f}")
                if logits .shape [-1 ]==11 :
                    dbg =torch .softmax (logits [0 ,0 ],dim =-1 )
                    log .info (f"   [debug] first_frame_softmax: {dbg .float ().cpu ().numpy ()}")
                else :
                    dbg =torch .sigmoid (logits [0 ,0 ])
                    log .info (f"   [debug] first_frame_sigmoid: {dbg .float ().cpu ().numpy ()}")

            T_l ,T_o =labels .shape [1 ],logits .shape [1 ]
            if T_l !=T_o :
                lbl_r =F .interpolate (labels .permute (0 ,2 ,1 ).float (),size =T_o ,mode ="nearest").permute (0 ,2 ,1 )
                nf_r =(n_frames .float ()*T_o /max (T_l ,1 )).long ().clamp (max =T_o )
            else :
                lbl_r ,nf_r =labels .float (),n_frames

            loss =pit_loss (logits ,lbl_r ,nf_r ,phase )
            n_samples =waveform .shape [0 ]
            split_loss +=loss .item ()*n_samples
            n_samples_total +=n_samples

            if device_type =="mps":
                logits_fp32 =model_der (batch ["waveform"].cpu ()).detach ().cpu ().float ()
            else :
                logits_fp32 =logits .detach ().cpu ().float ()

            T_l ,T_o =lbl_r .shape [1 ],logits_fp32 .shape [1 ]
            if T_l !=T_o :
                lbl_der =F .interpolate (lbl_r .cpu ().permute (0 ,2 ,1 ).float (),size =T_o ,mode ="nearest").permute (0 ,2 ,1 )
                nf_der =(nf_r .cpu ().float ()*T_o /max (T_l ,1 )).long ().clamp (max =T_o )
            else :
                lbl_der =lbl_r .cpu ().float ()
                nf_der =nf_r .cpu ()

            for thr in thr_grid :
                m =pit_der_batch (logits_fp32 ,lbl_der ,nf_der ,load_spk ,threshold =thr )
                der_acc [float (thr )]["miss"]+=m ["miss_frames"]
                der_acc [float (thr )]["fa"]+=m ["fa_frames"]
                der_acc [float (thr )]["conf"]+=m ["conf_frames"]
                der_acc [float (thr )]["total"]+=m ["total_frames"]

            n_val_crops +=1
            if max_crops >0 and n_val_crops >=max_crops :
                break

        avg_split_loss =split_loss /max (n_samples_total ,1 )
        der_by_thr ={}
        for thr in thr_grid :
            acc =der_acc [float (thr )]
            d =max (acc ["total"],1.0 )
            miss_thr =acc ["miss"]/d *100.0
            fa_thr =acc ["fa"]/d *100.0
            conf_thr =acc ["conf"]/d *100.0
            der_by_thr [f"{float (thr ):.2f}"]={
            "der":miss_thr +fa_thr +conf_thr ,
            "miss":miss_thr ,
            "fa":fa_thr ,
            "conf":conf_thr ,
            }

        base_key =f"{base_thr :.2f}"
        base_metrics =der_by_thr [base_key ]
        miss =base_metrics ["miss"]
        fa =base_metrics ["fa"]
        conf =base_metrics ["conf"]
        der =base_metrics ["der"]

        with torch .no_grad ():
            sm_all =decode_probs (logits_fp32 )
            hard_all =decode_probs_hard (logits_fp32 ,threshold =base_thr )
            max_p =sm_all .max ().item ()
            avg_active =hard_all .sum (-1 ).mean ().item ()
            log .info (f"   [val_debug] {split_name } | Max soft marginal: {max_p :.4f} | Avg active spk/frame: {avg_active :.2f}")

        all_results [split_name ]={
        "loss":avg_split_loss ,
        "der":der ,"miss":miss ,"fa":fa ,"conf":conf ,
        "der_sweep":der_by_thr ,
        }

        if src =="ALL"and spk =="ALL":
            total_val_loss =avg_split_loss
            total_val_der =der

        sweep_log =", ".join (
        f"thr={k }:{v ['der']:.2f}%"for k ,v in sorted (der_by_thr .items (),key =lambda kv :float (kv [0 ]))
        )
        log .info (
        f"  [val/{split_name }] ep{epoch } | "
        f"loss={avg_split_loss :.4f} | "
        f"DER@{base_thr :.2f}={der :.2f}% (M:{miss :.2f}% F:{fa :.2f}% C:{conf :.2f}%) | "
        f"sweep[{sweep_log }]"
        )

    log .info (
    f"  ★ VAL LOSS: {total_val_loss :.4f}  |  VAL DER (local, no collar): {total_val_der :.2f}%"
    f"\n  [NOTE] Local DER: crop-level, no collar — dùng để chọn best checkpoint."
    f"\n         Heavy DER: file-level, collar=0.25s — để đánh giá thực tế."
    f"\n         Hai số này không so sánh trực tiếp được."
    )

    return total_val_loss ,{
    "val_loss":total_val_loss ,
    "val_der":total_val_der ,
    "val_der_threshold":base_thr ,
    "val_der_sweep":all_results .get ("ALL/ALL",{}).get ("der_sweep",{}),
    "splits":all_results ,
    }

def _resample_probs_to_frame_grid (probs :np .ndarray ,actual_sec :float ,dt_sec :float =FRAME_SHIFT_SEC )->np .ndarray :
    target_frames =max (1 ,int (round (actual_sec /dt_sec )))
    if probs .shape [0 ]==target_frames :
        return probs .astype (np .float32 ,copy =False )
    x =torch .from_numpy (probs .T ).unsqueeze (0 ).float ()
    y =F .interpolate (x ,size =target_frames ,mode ="linear",align_corners =False )
    return y .squeeze (0 ).T .numpy ().astype (np .float32 ,copy =False )

def _align_chunk_probs (existing_overlap :np .ndarray ,current_overlap :np .ndarray )->np .ndarray :

    n_spk =existing_overlap .shape [1 ]
    if existing_overlap .shape [0 ]==0 or current_overlap .shape [0 ]==0 :
        return np .arange (n_spk ,dtype =int )
    T =min (existing_overlap .shape [0 ],current_overlap .shape [0 ])
    ex =existing_overlap [:T ]
    cu =current_overlap [:T ]
    cost =np .zeros ((n_spk ,n_spk ),dtype =np .float32 )
    for i in range (n_spk ):
        for j in range (n_spk ):
            cost [i ,j ]=np .mean (np .abs (ex [:,i ]-cu [:,j ]))
    row_ind ,col_ind =linear_sum_assignment (cost )
    perm =np .arange (n_spk ,dtype =int )
    for i ,j in zip (row_ind .tolist (),col_ind .tolist ()):
        perm [i ]=j
    return perm

def _binary_matrix_to_annotation (binary :np .ndarray ,uri :str ,dt_sec :float =FRAME_SHIFT_SEC ,min_dur_sec :float =0.10 ):
    from pyannote .core import Annotation ,Segment
    ann =Annotation (uri =uri )
    track_id =0
    T ,S =binary .shape
    for s in range (S ):
        active =binary [:,s ].astype (bool )
        in_seg =False
        seg_start =0
        for f in range (T +1 ):
            is_on =(f <T )and active [f ]
            if is_on and not in_seg :
                seg_start =f
                in_seg =True
            elif not is_on and in_seg :
                in_seg =False
                t_start =seg_start *dt_sec
                t_end =f *dt_sec
                if t_end -t_start >=min_dur_sec :
                    ann [Segment (t_start ,t_end ),f"H{track_id :06d}"]=f"SPK_{s :02d}"
                    track_id +=1
    return ann

def _reference_annotation_from_rttm (sample :Sample ):
    from pyannote .core import Annotation ,Segment
    ref =Annotation (uri =sample .wav_path )
    for idx ,seg in enumerate (parse_rttm (sample .rttm_path )):
        ref [Segment (seg ["start"],seg ["end"]),f"R{idx :06d}"]=seg ["speaker"]
    return ref

def _metric_breakdown (metric )->Dict [str ,float ]:
    der_pct =float (abs (metric ))*100.0
    miss_pct =fa_pct =conf_pct =-1.0
    try :
        detail =metric .report (display =False )
        def _find_col (df ,name :str ):
            for col in df .columns :
                label =col [-1 ]if isinstance (col ,tuple )else col
                if isinstance (label ,str )and label .lower ()==name .lower ():
                    return col
            return None
        last =detail .iloc [-1 ]
        c_total =_find_col (detail ,"total")
        c_miss =_find_col (detail ,"missed detection")
        c_fa =_find_col (detail ,"false alarm")
        c_conf =_find_col (detail ,"confusion")
        if c_total is not None :
            total_dur =float (last [c_total ])
            if total_dur >0 :
                if c_miss is not None :miss_pct =float (last [c_miss ])/total_dur *100.0
                if c_fa is not None :fa_pct =float (last [c_fa ])/total_dur *100.0
                if c_conf is not None :conf_pct =float (last [c_conf ])/total_dur *100.0
    except Exception as e :
        log .warning (f"  [heavy_valid] metric breakdown failed: {e }")
    return {"der":der_pct ,"miss":miss_pct ,"fa":fa_pct ,"conf":conf_pct }

@torch .no_grad ()
def heavy_file_level_der_eval (
model :DiariZenWrapper ,
samples :List [Sample ],
cfg :FinetuneConfig ,
device :torch .device ,
device_type :str ,
threshold :float =0.5 ,
collar :float =0.25 ,
split_name :str ="heavy",
)->Dict [str ,float ]:
    try :
        from pyannote .metrics .diarization import DiarizationErrorRate
    except ImportError :
        log .warning ("[heavy_valid] pyannote.metrics not installed — skipping.")
        return {"der":-1.0 ,"miss":-1.0 ,"fa":-1.0 ,"conf":-1.0 ,"n_files":0 ,"threshold":threshold }

    metric =DiarizationErrorRate (collar =collar ,skip_overlap =False )
    model .eval ()
    chunk_smp =int (cfg .chunk_sec *SR )
    hop_smp =int (getattr (cfg ,"heavy_chunk_hop_sec",cfg .chunk_sec )*SR )
    hop_smp =max (1 ,min (hop_smp ,chunk_smp ))
    n_files_ok =0

    for sample in samples :
        if sample .n_speakers >cfg .max_speakers :
            continue
        try :
            wav ,sr =_load_audio (sample .wav_path )
            if wav .shape [0 ]>1 :
                wav =wav .mean (0 ,keepdim =True )
            if sr !=SR :
                wav =torchaudio .functional .resample (wav ,sr ,SR )
        except Exception as e :
            log .warning (f"  [heavy_valid/{split_name }] load failed {sample .wav_path }: {e }")
            continue

        total_smp =wav .shape [1 ]
        total_frames =max (1 ,int (round (total_smp /SR /FRAME_SHIFT_SEC )))
        sum_probs =np .zeros ((total_frames ,cfg .max_speakers ),dtype =np .float32 )
        cnt_probs =np .zeros ((total_frames ,1 ),dtype =np .float32 )

        for start in range (0 ,total_smp ,hop_smp ):
            end =min (start +chunk_smp ,total_smp )
            actual_smp =end -start
            if actual_smp <SR :
                break
            chunk =wav [:,start :end ]
            if actual_smp <chunk_smp :
                chunk =F .pad (chunk ,(0 ,chunk_smp -actual_smp ))
            chunk =chunk .to (device ,non_blocking =(device_type =="cuda"))
            with _autocast (cfg .fp16 and device_type =="cuda",device_type ,getattr (cfg ,"amp_dtype","float16")):
                logits =model (chunk .unsqueeze (0 ))[0 ].detach ().cpu ().float ()
            probs =decode_probs (logits ,n_spk =cfg .max_speakers ).cpu ().numpy ().astype (np .float32 )
            probs =_resample_probs_to_frame_grid (probs ,actual_smp /SR ,FRAME_SHIFT_SEC )

            start_fr =int (round (start /SR /FRAME_SHIFT_SEC ))
            end_fr =min (total_frames ,start_fr +probs .shape [0 ])
            probs =probs [:max (0 ,end_fr -start_fr )]
            if probs .shape [0 ]==0 :
                continue

            ov_start =start_fr
            ov_end =min (end_fr ,start_fr +int (round ((chunk_smp -hop_smp )/SR /FRAME_SHIFT_SEC )))
            if ov_end >ov_start and np .any (cnt_probs [ov_start :ov_end ,0 ]>0 ):
                existing =sum_probs [ov_start :ov_end ]/np .clip (cnt_probs [ov_start :ov_end ],1e-8 ,None )
                current =probs [:ov_end -ov_start ]
                perm =_align_chunk_probs (existing ,current )
                probs =probs [:,perm ]

            sum_probs [start_fr :end_fr ]+=probs
            cnt_probs [start_fr :end_fr ]+=1.0

        if np .all (cnt_probs ==0 ):
            log .warning (f"  [heavy_valid/{split_name }] no chunk coverage for {sample .wav_path }")
            continue

        avg_probs =sum_probs /np .clip (cnt_probs ,1e-8 ,None )
        hyp_bin =(avg_probs >threshold ).astype (np .uint8 )
        hyp =_binary_matrix_to_annotation (hyp_bin ,uri =sample .wav_path )
        ref =_reference_annotation_from_rttm (sample )
        if len (ref )==0 :
            log .warning (f"  [heavy_valid/{split_name }] empty ref for {sample .wav_path }")
            continue
        try :
            metric (ref ,hyp )
            n_files_ok +=1
        except Exception as e :
            log .warning (f"  [heavy_valid/{split_name }] eval failed {sample .wav_path }: {e }")

    if n_files_ok ==0 :
        return {"der":-1.0 ,"miss":-1.0 ,"fa":-1.0 ,"conf":-1.0 ,"n_files":0 ,"threshold":threshold }

    out =_metric_breakdown (metric )
    out .update ({"n_files":n_files_ok ,"threshold":float (threshold )})
    return out

@torch .no_grad ()
def run_heavy_validation (
model :DiariZenWrapper ,
heavy_splits :Dict [str ,List [Sample ]],
cfg :FinetuneConfig ,
device :torch .device ,
device_type :str ,
epoch :int ,
)->Dict [str ,Dict ]:
    if not getattr (cfg ,"heavy_validate_enabled",False ):
        return {}
    out ={}
    thr =float (getattr (cfg ,"heavy_der_threshold",0.50 ))
    log .info (f"  [heavy_valid] epoch={epoch } | threshold={thr :.2f}")
    for split_name ,samples in heavy_splits .items ():
        if not samples :
            out [split_name ]={"der":-1.0 ,"miss":-1.0 ,"fa":-1.0 ,"conf":-1.0 ,"n_files":0 ,"threshold":thr }
            log .info (f"  [heavy_valid/{split_name }] skipped (0 files)")
            continue
        res =heavy_file_level_der_eval (model ,samples ,cfg ,device ,device_type ,threshold =thr ,split_name =split_name )
        out [split_name ]=res
        log .info (
        f"  [heavy_valid/{split_name }] n_files={res ['n_files']} | DER={res ['der']:.2f}% "
        f"(M:{res ['miss']:.2f}% F:{res ['fa']:.2f}% C:{res ['conf']:.2f}%)"
        )
    return out

def train_phase (
model :DiariZenWrapper ,
train_files :List [Sample ],
val_registry :ValRegistry ,
heavy_splits :Dict [str ,List [Sample ]],
cfg :FinetuneConfig ,
phase :PhaseConfig ,
device :torch .device ,
device_type :str ,
ecapa ,
out_dir :Path ,
global_epoch :int ,
augmenter :Optional [AudioAugmenter ],
)->Tuple [int ,float ,List [Dict ],List [Dict ]]:

    log .info (f"\n{'═'*70 }")
    log .info (f"  PHASE: {phase .name }  ({phase .epochs } epochs, freeze={phase .freeze_strategy })")
    log .info (f"{'═'*70 }")

    apply_freeze (model ,phase .freeze_strategy ,phase .unfreeze_top_n )
    optimizer =build_optimizer (model ,phase )

    from ft_dataloader import sliding_window_samples
    _mix0 =phase .data_mix .get (0 )
    _active =[s for s in train_files if _mix0 .get (s .source ,0.0 )>0.0 ]or train_files
    _phase_stride0 =cfg .train_stride_sec_p3 if "phase3"in phase .name .lower ()else cfg .train_stride_sec
    _crops0 =sliding_window_samples (_active ,cfg .chunk_sec ,_phase_stride0 )
    _steps_ep0 =max (len (_crops0 )//cfg .batch_size ,1 )
    total_steps =_steps_ep0 *phase .epochs
    del _crops0 ,_active
    scheduler =WarmupCosine (optimizer ,warmup =phase .warmup_steps ,total =total_steps )
    amp_dtype =getattr (cfg ,"amp_dtype","float16")
    scaler =_make_scaler (cfg .fp16 ,device_type ,amp_dtype )
    trainable_params =[p for p in model .parameters ()if p .requires_grad ]
    use_non_blocking =device_type =="cuda"
    optimizer .zero_grad (set_to_none =True )

    best_val_loss =float ("inf")
    best_val_der =float ("inf")
    patience_cnt =0
    history =[]
    heavy_history =[]

    model_cpu_ref =None
    if device_type =="mps":
        from diarizen .pipelines .inference import DiariZenPipeline as _DZP2
        _pl2 =_DZP2 .from_pretrained (cfg .pretrained_model )
        model_cpu_ref =DiariZenWrapper (_pl2 ,max_speakers =4 ).cpu ().eval ()
        log .info ("  [MPS] CPU shadow model created for accurate val DER")

    for ep_idx in range (phase .epochs ):
        global_epoch +=1
        phase_stride =cfg .train_stride_sec_p3 if "phase3"in phase .name .lower ()else cfg .train_stride_sec
        loader =build_train_loader (
        train_files ,phase ,ep_idx ,cfg ,
        augmenter if phase .use_augmentation else None ,
        phase_stride ,
        )

        model .train ()
        epoch_loss =0.0
        n_samples_train =0
        n_steps =0

        log .info (f"\n{'─'*60 }")
        log .info (f"  [{phase .name }] Epoch {ep_idx +1 }/{phase .epochs }  (global ep {global_epoch })")
        log .info (f"  mix: {phase .data_mix .get (ep_idx )}")
        log .info (f"{'─'*60 }")

        pbar =tqdm (loader ,desc =f"Ep {ep_idx +1 }/{phase .epochs }",leave =True ,dynamic_ncols =True )
        for step ,batch in enumerate (pbar ):
            waveform =batch ["waveform"].to (device ,non_blocking =use_non_blocking )
            labels =batch ["labels"].to (device ,non_blocking =use_non_blocking )
            n_frames =batch ["n_frames"].to (device ,non_blocking =use_non_blocking )

            with _autocast (cfg .fp16 and device_type =="cuda",device_type ,amp_dtype ):
                logits =model (waveform )
                T_l ,T_o =labels .shape [1 ],logits .shape [1 ]
                if T_l !=T_o :
                    lbl_r =F .interpolate (labels .permute (0 ,2 ,1 ).float (),size =T_o ,mode ="nearest").permute (0 ,2 ,1 )
                    nf_r =(n_frames .float ()*T_o /max (T_l ,1 )).long ().clamp (max =T_o )
                else :
                    lbl_r ,nf_r =labels .float (),n_frames
                loss =pit_loss (logits ,lbl_r ,nf_r ,phase )/cfg .grad_accum

            if scaler .is_enabled ():
                scaler .scale (loss ).backward ()
            else :
                loss .backward ()

            if (step +1 )%cfg .grad_accum ==0 :
                if scaler .is_enabled ():
                    scaler .unscale_ (optimizer )
                nn .utils .clip_grad_norm_ (trainable_params ,phase .grad_clip )
                if scaler .is_enabled ():
                    scaler .step (optimizer )
                    scaler .update ()
                else :
                    optimizer .step ()
                optimizer .zero_grad (set_to_none =True )
                scheduler .step ()

            epoch_loss +=loss .item ()*cfg .grad_accum *waveform .shape [0 ]
            n_samples_train +=waveform .shape [0 ]
            n_steps +=1
            pbar .set_postfix (loss =f"{loss .item ()*cfg .grad_accum :.4f}",lr =f"{scheduler .get_lr ()[0 ]:.2e}")

        remainder =n_steps %cfg .grad_accum
        if remainder >0 :
            if scaler .is_enabled ():
                scaler .unscale_ (optimizer )
            nn .utils .clip_grad_norm_ (trainable_params ,phase .grad_clip )
            if scaler .is_enabled ():
                scaler .step (optimizer )
                scaler .update ()
            else :
                optimizer .step ()
            optimizer .zero_grad (set_to_none =True )
            scheduler .step ()

        avg_loss =epoch_loss /max (n_samples_train ,1 )
        log .info (f"\n  avg_train_loss={avg_loss :.4f}")

        val_metrics :Dict ={}
        do_val =(ep_idx +1 )%cfg .val_every_n_epochs ==0 or (ep_idx +1 )==phase .epochs
        if do_val :
            val_loss ,val_metrics =run_validation (
            model ,val_registry ,cfg ,phase ,device ,device_type ,
            ecapa ,global_epoch ,model_cpu_ref =model_cpu_ref
            )

            val_der =val_metrics .get ("val_der",float ("inf"))
            if val_der <best_val_der :
                best_val_der =val_der
                best_val_loss =val_loss
                patience_cnt =0
                best_path =str (out_dir /"best_model.pth")
                clean_sd ={k [11 :]if k .startswith ("_seg_model.")else k :v for k ,v in model .state_dict ().items ()}
                torch .save ({
                "epoch":global_epoch ,
                "val_loss":val_loss ,"val_der":val_der ,"model_state":clean_sd ,
                },best_path )
                log .info (f"  ★ New best val_der={val_der :.2f}% (loss={val_loss :.4f}) → {best_path }")
            else :
                patience_cnt +=1
                log .info (f"  No improvement. Patience: {patience_cnt }/{cfg .patience }")
                if patience_cnt >=cfg .patience :
                    log .info (f"  Early stopping at epoch {global_epoch }")
                    break

        heavy_metrics :Dict ={}
        do_heavy =(
        getattr (cfg ,"heavy_validate_enabled",False )
        and bool (heavy_splits )
        and (global_epoch %max (1 ,int (getattr (cfg ,"heavy_validate_every",3 )))==0 or (ep_idx +1 )==phase .epochs )
        )
        if do_heavy :
            heavy_metrics =run_heavy_validation (model ,heavy_splits ,cfg ,device ,device_type ,global_epoch )
            heavy_history .append ({
            "global_epoch":global_epoch ,
            "phase":phase .name ,
            "threshold":float (getattr (cfg ,"heavy_der_threshold",0.50 )),
            "splits":heavy_metrics ,
            })

        if (ep_idx +1 )%cfg .save_every_n_epochs ==0 :
            clean_sd ={k [11 :]if k .startswith ("_seg_model.")else k :v for k ,v in model .state_dict ().items ()}
            ep_path =str (out_dir /f"ep{global_epoch :03d}_{phase .name }.pth")
            torch .save ({
            "epoch":global_epoch ,"phase":phase .name ,
            "train_loss":avg_loss ,**val_metrics ,
            "heavy_valid":heavy_metrics ,
            "model_state":clean_sd ,
            },ep_path )
            log .info (f"  [ckpt] {ep_path }")

        history .append ({"global_epoch":global_epoch ,"phase":phase .name ,
        "train_loss":avg_loss ,**val_metrics ,
        "heavy_valid":heavy_metrics })

    return global_epoch ,best_val_der ,history ,heavy_history

def main ():
    import random
    cfg =FinetuneConfig ()

    if TEST_MODE :
        MAX_N_FILES =2
        log .info ("="*60 )
        log .info ("  MODE: SMOKE TEST  (TEST_MODE=True trong ft_config.py)")
        log .info (f"  max_n_files={MAX_N_FILES } | chunk={cfg .chunk_sec }s | batch={cfg .batch_size } | phases=1")
        log .info ("="*60 )
    else :
        MAX_N_FILES =None
        log .info ("="*60 )
        log .info ("  MODE: FULL TRAINING  (TEST_MODE=False)")
        log .info ("="*60 )

    random .seed (cfg .seed );np .random .seed (cfg .seed );torch .manual_seed (cfg .seed )

    device =torch .device (
    "cuda"if torch .cuda .is_available ()else
    "mps"if torch .backends .mps .is_available ()else "cpu"
    )
    device_type =_get_device_type (device )
    if cfg .require_cuda and device_type !="cuda":
        log .error ("CUDA is required for this recipe. Refusing to fall back to MPS/CPU.")
        raise SystemExit (1 )
    log .info (f"Device: {device }")

    if device_type =="cuda":
        if hasattr (torch ,"set_float32_matmul_precision"):
            torch .set_float32_matmul_precision (str (getattr (cfg ,"matmul_precision","high")))
        if getattr (cfg ,"enable_tf32",True ):
            torch .backends .cuda .matmul .allow_tf32 =True
            torch .backends .cudnn .allow_tf32 =True
        if getattr (cfg ,"cudnn_benchmark",True ):
            torch .backends .cudnn .benchmark =True
        log .info (
        f"  [CUDA] amp_dtype={getattr (cfg ,'amp_dtype','float16')} | "
        f"TF32={getattr (cfg ,'enable_tf32',True )} | "
        f"cudnn_benchmark={getattr (cfg ,'cudnn_benchmark',True )}"
        )

    if device_type =="mps"and cfg .fp16 :
        log .warning ("  [!] MPS detected. Disabling AMP FP16 to prevent internal NaN/Infs in WavLM convolution blocks.")
        cfg .fp16 =False

    log .info ("\n[Data] Scanning...")

    train_files =scan_all_sources (cfg ,max_n_files_per_source =MAX_N_FILES )
    if not train_files :
        log .error ("No training files found!");sys .exit (1 )

    for s in train_files :
        enroll_ok ="\u2713"if s .enrollment_dir else "\u2717"
        log .info (f"  [train/{s .source }] {Path (s .wav_path ).parent .parent .name }"
        f" | dur={s .duration :.1f}s | n_spk={s .n_speakers }"
        f" | enroll={enroll_ok }")

    from pathlib import Path as _P
    val_files_labeled =scan_fixed_split_dir (
    VAL_LABELED_DIR ,
    str (_P (VAL_LABELED_DIR )/"metadata.txt"),
    max_speakers =cfg .max_speakers ,
    source_name ="val_labeled",
    )
    val_files_datamix =scan_fixed_split_dir (
    VAL_DATA_DIR ,
    str (_P (VAL_DATA_DIR )/"metadata.txt"),
    max_speakers =cfg .max_speakers ,
    source_name ="val_datamix",
    )

    from ft_dataloader import stratified_sample_by_overlap ,stratified_sample_by_overlap_and_speakers

    if len (val_files_labeled )>=cfg .max_val_files :

        val_files =stratified_sample_by_overlap_and_speakers (val_files_labeled ,cfg .max_val_files )
    else :
        needed =cfg .max_val_files -len (val_files_labeled )
        sampled_mix =stratified_sample_by_overlap (val_files_datamix ,needed )
        val_files =val_files_labeled +sampled_mix

    if not val_files :
        log .warning ("[Val] No val files found — using train subset for val (TEST_MODE safety)")
        val_files =list (train_files [:cfg .max_val_files ])

    log .info (f"\n[Val] final_set={len (val_files )} (labeled={len (val_files_labeled )}, sampled_mix={len (val_files )-len (val_files_labeled )})")

    test_files_labeled =scan_fixed_split_dir (
    TEST_LABELED_DIR ,
    str (_P (TEST_LABELED_DIR )/"metadata.txt"),
    max_speakers =cfg .max_speakers ,
    source_name ="test_labeled",
    )
    test_files_datamix =scan_fixed_split_dir (
    TEST_DATA_DIR ,
    str (_P (TEST_DATA_DIR )/"metadata.txt"),
    max_speakers =cfg .max_speakers ,
    source_name ="test_datamix",
    )
    test_files =test_files_labeled +test_files_datamix
    if not test_files :
        log .warning ("[Test] No test files found — using val files for final eval")
        test_files =list (val_files )
    log .info (f"[Test] test_labeled={len (test_files_labeled )} | test_datamix={len (test_files_datamix )} | total={len (test_files )}")

    augmenter =AudioAugmenter (sr =SR )

    val_registry =ValRegistry (val_files ,cfg )

    def _limit_heavy (files :List [Sample ])->List [Sample ]:
        keep =[s for s in files if s .n_speakers <=cfg .max_speakers ]
        max_keep =int (getattr (cfg ,"heavy_max_files_per_split",0 ))
        if max_keep >0 :
            keep =keep [:max_keep ]
        return keep

    heavy_splits ={
    "heavy_labeled":_limit_heavy (val_files_labeled ),
    "heavy_datamix":_limit_heavy (val_files_datamix ),
    }
    for _name ,_files in heavy_splits .items ():
        log .info (f"[Heavy] {_name }={len (_files )} files (<= {cfg .max_speakers } speakers)")

    ecapa =load_ecapa_model ()

    log .info (f"\n[Model] Loading: {cfg .pretrained_model }")
    try :
        import torchaudio as _torchaudio
        if not hasattr (_torchaudio ,"list_audio_backends"):
            _torchaudio .list_audio_backends =lambda :[]

        stored_safe_globals =None
        safe_globals =[torch .torch_version .TorchVersion ]
        try :
            from pyannote .audio .core .task import Problem ,Resolution ,Specifications ,Task
            safe_globals .extend ([Problem ,Resolution ,Specifications ,Task ])
        except Exception :
            pass

        if hasattr (torch ,"serialization")and hasattr (torch .serialization ,"add_safe_globals"):
            if hasattr (torch .serialization ,"get_safe_globals"):
                stored_safe_globals =torch .serialization .get_safe_globals ()
            torch .serialization .add_safe_globals (safe_globals )

        from diarizen .pipelines .inference import DiariZenPipeline
        pipeline =DiariZenPipeline .from_pretrained (cfg .pretrained_model )
        log .info ("  DiariZenPipeline ")
    except (ImportError ,RuntimeError ,Exception )as e :

        log .error (f"Failed to load DiariZenPipeline: {type (e ).__name__ }: {e }")
        log .error ("  → Thử: pip install transformers accelerate")
        raise SystemExit (1 )
    finally :
        if hasattr (torch ,"serialization")and hasattr (torch .serialization ,"clear_safe_globals"):
            torch .serialization .clear_safe_globals ()
            if stored_safe_globals :
                torch .serialization .add_safe_globals (stored_safe_globals )

    model =DiariZenWrapper (pipeline ,max_speakers =cfg .max_speakers ).to (device )
    if device_type =="cuda"and getattr (cfg ,"compile_model",False )and hasattr (torch ,"compile"):
        try :
            model ._seg_model =torch .compile (
            model ._seg_model ,
            mode =str (getattr (cfg ,"compile_mode","max-autotune")),
            dynamic =True ,
            )
            log .info (f"  [CUDA] torch.compile enabled: mode={getattr (cfg ,'compile_mode','max-autotune')}")
        except Exception as e :
            log .warning (f"  [CUDA] torch.compile unavailable/failed: {e }")

    out_dir =Path (cfg .output_dir )
    out_dir .mkdir (parents =True ,exist_ok =True )
    log .info (f"Output: {out_dir }")

    phase_configs =build_phase_configs ()
    global_epoch =0
    full_history =[]
    heavy_history =[]

    best_ckpt_path =str (out_dir /"best_model.pth")
    for phase in phase_configs :
        global_epoch ,phase_best_der ,phase_history ,phase_heavy_history =train_phase (
        model ,train_files ,val_registry ,heavy_splits ,cfg ,phase ,
        device ,device_type ,ecapa ,out_dir ,global_epoch ,augmenter ,
        )
        full_history .extend (phase_history )
        heavy_history .extend (phase_heavy_history )
        log .info (f"\n  [Phase done] {phase .name } | best_val_der={phase_best_der :.2f}%")

    log .info (f"\n{'═'*70 }")
    log .info ("TRAINING COMPLETE")
    log .info (f"{'═'*70 }")
    for r in full_history :
        val_loss =r .get ("val_loss",-1.0 )
        log .info (
        f"  ep{r ['global_epoch']:>3} [{r ['phase']:<20}] "
        f"train_loss={r ['train_loss']:.4f}"
        f"  val_loss={val_loss :.4f}"
        if val_loss >=0 else
        f"  ep{r ['global_epoch']:>3} [{r ['phase']:<20}] "
        f"train_loss={r ['train_loss']:.4f}  (no val this epoch)"
        )

    out_json =str (out_dir /"training_history.json")
    with open (out_json ,"w",encoding ="utf-8")as f :
        json .dump (full_history ,f ,indent =2 ,ensure_ascii =False )
    log .info (f"\nHistory saved → {out_json }")

    heavy_json =str (out_dir /"heavy_validation_history.json")
    with open (heavy_json ,"w",encoding ="utf-8")as f :
        json .dump (heavy_history ,f ,indent =2 ,ensure_ascii =False )
    log .info (f"Heavy history saved → {heavy_json }")

    if Path (best_ckpt_path ).exists ()and test_files :
        log .info (f"\n{'═'*70 }")
        log .info ("  FINAL HEAVY EVALUATION  (test_labeled + test_datamix)")
        log .info (f"  Checkpoint: {best_ckpt_path }")
        log .info (f"  Metric: file-level DER, collar=0.25s, Hungarian speaker alignment")
        log .info (f"{'═'*70 }")

        ckpt =torch .load (best_ckpt_path ,map_location =device ,weights_only =False )
        model_state =ckpt ["model_state"]
        if any (k .startswith ("_seg_model.")for k in model_state .keys ()):
            model .load_state_dict (model_state )
        else :
            model ._seg_model .load_state_dict (model_state ,strict =False )
        model .eval ()

        final_test_splits ={
        "test_labeled":[s for s in test_files_labeled if s .n_speakers <=cfg .max_speakers ],
        "test_datamix":[s for s in test_files_datamix if s .n_speakers <=cfg .max_speakers ],
        }
        final_results ={}
        for split_name ,split_files in final_test_splits .items ():
            if not split_files :
                log .info (f"  [final_eval/{split_name }] skipped (0 files)")
                final_results [split_name ]={"der":-1.0 ,"n_files":0 }
                continue
            res =heavy_file_level_der_eval (
            model ,split_files ,cfg ,device ,device_type ,
            threshold =cfg .heavy_der_threshold ,
            collar =0.25 ,
            split_name =split_name ,
            )
            final_results [split_name ]=res
            log .info (
            f"  [final_eval/{split_name }] n_files={res ['n_files']} | "
            f"DER={res ['der']:.2f}% (M:{res ['miss']:.2f}% F:{res ['fa']:.2f}% C:{res ['conf']:.2f}%)"
            )

        final_path =str (out_dir /"final_evaluation.json")
        with open (final_path ,"w",encoding ="utf-8")as f :
            json .dump (final_results ,f ,indent =2 ,ensure_ascii =False )
        log .info (f"Final evaluation saved → {final_path }")
    else :
        if not Path (best_ckpt_path ).exists ():
            log .warning ("best_model.pth không tồn tại — bỏ qua final evaluation")
        if not test_files :
            log .warning ("Không có test files — bỏ qua final evaluation")

if __name__ =="__main__":
    main ()