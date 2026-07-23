import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.eval_asr_from_npz import eval_asr_from_npz
# from eval.eval_fid_from_npz import eval_fid_from_npz
# from eval.eval_no_ref_quality_from_npz import eval_no_ref_quality_from_npz

def eval_all(
    file_name=None,
    re_logger=True,
    quant=False,
    model_name=None,
    use_npz_source_query=True,
    run_quality=True,
    run_asr=True,
    run_fid=False,
):
    """Evaluate an NPZ directory, with quality, ASR, and FID independently runnable."""
    results = {}
    if run_quality:
        # LPIPS is an optional evaluation dependency, so import it only when used.
        from eval.eval_quality_from_npz import eval_quality_from_npz

        results["quality"] = eval_quality_from_npz(
            file_name=file_name,
            re_logger=re_logger,
            quant=quant,
        )

    if run_fid:
        from eval.eval_fid_from_npz import eval_fid_from_npz

        results["fid"] = eval_fid_from_npz(
            file_name=file_name,
            re_logger=re_logger,
            quant=quant,
        )

    if run_asr:
        if model_name is None:
            raise ValueError("model_name is required when run_asr=True")
        results["asr"] = eval_asr_from_npz(
            file_name=file_name,
            re_logger=re_logger,
            quant=quant,
            model_name=model_name,
            use_npz_source_query=use_npz_source_query,
            use_prompt_label_as_correct=(
                "ASA" in str(file_name) or "nips2017" in str(file_name)
            ),
        )
    # eval_no_ref_quality_from_npz(file_name=file_name, re_logger=re_logger, quant=quant)
    return results


if __name__ == "__main__":
    dir_list = [
        # "outputs/nips2017/resnet50/re_vlm_alter",
        # "outputs/nips2017/swin/re_vlm_alter",
        # "outputs/nips2017/convnext/re_vlm_alter",
        # "outputs/nips2017/vim-small/re_vlm_alter",
        # "outputs/nips2017/deit/re_vlm_alter",
        # "outputs/nips2017/mambavision/re_vlm_alter",
        # "outputs/nips2017/adv_res/re_vlm_alter",
        # "outputs/nips2017/adv_inc/re_vlm_alter",
        # "outputs/nips2017/resnet50/re_greedy_alter",
        # "outputs/nips2017/swin/re_greedy_alter",
        # "outputs/nips2017/convnext/re_greedy_alter",
        # "outputs/nips2017/vim-small/re_greedy_alter",
        # "outputs/nips2017/deit/re_greedy_alter",
        # "outputs/nips2017/mambavision/re_greedy_alter",
        # "outputs/nips2017/adv_res/re_greedy_alter",
        # "outputs/nips2017/adv_inc/re_greedy_alter",
        # "/vgg19/cwor_attack_q100",
        # "/resnet50/cwor_attack_q100",
        "outputs/nips2017/resnet50/flux2_and_attack_q100_eval",
    ]
    models = [
        "resnet50",
        "wrn50",
        "inception_v3", 
        "convnext",
        "vgg19",
        "vit",
        "swin",
        "deit",
        "vim-small",
        "mambavision",
        "adv_inc",
        "adv_res",

    ]
    for dir in dir_list:
        print("*********************** Eval All {} ***********************".format(dir))

        '''Eval normal quanted (8-bit image) results for AdvAD (quant=True)'''
        eval_all(file_name=dir, re_logger=True, quant=True, model_name=models)

        '''Eval raw floating-point data w/o quant for AdvAD-X (quant=False)'''
        # eval_all(file_name=dir, re_logger=True, quant=False, model_name="resnet50")

        print("************************ Done ************************")
