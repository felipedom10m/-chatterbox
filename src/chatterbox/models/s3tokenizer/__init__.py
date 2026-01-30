from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)


SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1



def drop_invalid_tokens(x):
    """Drop SoS and EoS"""
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"
    # Ensure we are working with a 1D tensor and safe integer indices
    if len(x.shape) == 2:
        x = x[0]

    if SOS in x:
        s_idx = (x == SOS).nonzero(as_tuple=True)[0]
        s = int(s_idx[0].item()) + 1 if s_idx.numel() > 0 else 0
    else:
        s = 0

    if EOS in x:
        e_idx = (x == EOS).nonzero(as_tuple=True)[0]
        e = int(e_idx[0].item()) if e_idx.numel() > 0 else None
    else:
        e = None

    try:
        x = x[s: e]
    except Exception:
        # Fallback: return original tokens if slicing fails
        return x
    return x
