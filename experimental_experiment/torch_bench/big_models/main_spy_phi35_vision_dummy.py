"""
From `microsoft/Phi-3.5-vision-instruct
<https://huggingface.co/microsoft/Phi-3.5-vision-instruct>`_

-- load processor from 'microsoft/Phi-3.5-vision-instruct'
/home/xadupre/vv/this/lib/python3.10/site-packages/transformers/models/auto/image_processing_auto.py:520: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
  warnings.warn(
-- download image from 'https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg'
   size: (2048, 1152)
-- create inputs
-- types: BatchFeature(data=dict(input_ids:T7s1x777[-1:32010],attention_mask:T7s1x777[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672]))
-- image_sizes tensor([[672, 672]], device='cuda:0')
-- intercept forward
-- inputs type: BatchFeature(data=dict(input_ids:T7r2,attention_mask:T7r2,pixel_values:T1r5,image_sizes:T7r2))
        transformers/generation/configuration_utils.py:590: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
`get_max_cache()` is deprecated for all Cache classes. Use `get_max_cache_shape()` instead. Calling `get_max_cache()` will raise error from v4.48
forward input:
args: ()
kwargs: dict(input_ids:T7s1x777[-1:32010],position_ids:T7s1x777[0:776],past_key_values:DynamicCache(key_cache=#0[], DynamicCache(value_cache=#0[]),use_cache:int[True],attention_mask:T7s1x777[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[    1, 32010, 29871,    13,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
            -1,     1, 29871,    13, 11139,  3034,   675,   278, 19810,   310,
          2243,  2247, 29889, 32007, 29871,    13, 32001]], device='cuda:0')
WARNING:experimental_experiment.torch_models.fromhub.modeling_phi3_v:You are not running the flash-attention implementation, expect numerical differences.
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[22751:22751],position_ids:T7s1x1[777:777],past_key_values:DynamicCache(key_cache=#1[T1s1x32x777x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x777x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x778[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[22751]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27448:27448],position_ids:T7s1x1[778:778],past_key_values:DynamicCache(key_cache=#1[T1s1x32x778x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x778x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x779[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27448]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[779:779],past_key_values:DynamicCache(key_cache=#1[T1s1x32x779x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x779x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x780[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[780:780],past_key_values:DynamicCache(key_cache=#1[T1s1x32x780x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x780x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x781[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[781:781],past_key_values:DynamicCache(key_cache=#1[T1s1x32x781x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x781x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x782[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[782:782],past_key_values:DynamicCache(key_cache=#1[T1s1x32x782x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x782x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x783[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27362:27362],position_ids:T7s1x1[783:783],past_key_values:DynamicCache(key_cache=#1[T1s1x32x783x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x783x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x784[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27362]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[784:784],past_key_values:DynamicCache(key_cache=#1[T1s1x32x784x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x784x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x785[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[785:785],past_key_values:DynamicCache(key_cache=#1[T1s1x32x785x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x785x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x786[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[786:786],past_key_values:DynamicCache(key_cache=#1[T1s1x32x786x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x786x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x787[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[787:787],past_key_values:DynamicCache(key_cache=#1[T1s1x32x787x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x787x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x788[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[788:788],past_key_values:DynamicCache(key_cache=#1[T1s1x32x788x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x788x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x789[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[789:789],past_key_values:DynamicCache(key_cache=#1[T1s1x32x789x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x789x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x790[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27362:27362],position_ids:T7s1x1[790:790],past_key_values:DynamicCache(key_cache=#1[T1s1x32x790x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x790x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x791[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27362]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[791:791],past_key_values:DynamicCache(key_cache=#1[T1s1x32x791x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x791x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x792[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[792:792],past_key_values:DynamicCache(key_cache=#1[T1s1x32x792x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x792x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x793[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27362:27362],position_ids:T7s1x1[793:793],past_key_values:DynamicCache(key_cache=#1[T1s1x32x793x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x793x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x794[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27362]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[794:794],past_key_values:DynamicCache(key_cache=#1[T1s1x32x794x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x794x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x795[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[795:795],past_key_values:DynamicCache(key_cache=#1[T1s1x32x795x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x795x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x796[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27362:27362],position_ids:T7s1x1[796:796],past_key_values:DynamicCache(key_cache=#1[T1s1x32x796x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x796x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x797[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27362]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[797:797],past_key_values:DynamicCache(key_cache=#1[T1s1x32x797x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x797x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x798[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[798:798],past_key_values:DynamicCache(key_cache=#1[T1s1x32x798x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x798x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x799[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[799:799],past_key_values:DynamicCache(key_cache=#1[T1s1x32x799x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x799x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x800[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[800:800],past_key_values:DynamicCache(key_cache=#1[T1s1x32x800x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x800x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x801[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[801:801],past_key_values:DynamicCache(key_cache=#1[T1s1x32x801x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x801x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x802[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[802:802],past_key_values:DynamicCache(key_cache=#1[T1s1x32x802x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x802x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x803[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27362:27362],position_ids:T7s1x1[803:803],past_key_values:DynamicCache(key_cache=#1[T1s1x32x803x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x803x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x804[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27362]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[7369:7369],position_ids:T7s1x1[804:804],past_key_values:DynamicCache(key_cache=#1[T1s1x32x804x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x804x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x805[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[7369]], device='cuda:0')
forward input:
args: ()
kwargs: dict(input_ids:T7s1x1[27718:27718],position_ids:T7s1x1[805:805],past_key_values:DynamicCache(key_cache=#1[T1s1x32x805x96[-5.475561141967773:6.331610202789307]], DynamicCache(value_cache=#1[T1s1x32x805x96[-5.408420562744141:4.858865261077881]]),use_cache:int[True],attention_mask:T7s1x806[1:1],pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],image_sizes:T7s1x2[672:672],return_dict:int[True])
tensor([[27718]], device='cuda:0')
"""

from PIL import Image
import requests
from transformers import AutoProcessor
from experimental_experiment.helpers import string_type
from experimental_experiment.torch_models.llm_model_helper import get_phi35_vision_instruct

model, *_ = get_phi35_vision_instruct(num_hidden_layers=1)
model = model.to("cuda")

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
model_id = "microsoft/Phi-3.5-vision-instruct"
print(f"-- load processor from {model_id!r}")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)

images = []
placeholder = ""

# Note: if OOM, you might consider reduce number of frames in this example.
for i in range(1, 1):
    url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
    print(f"-- download image from {url!r}")
    img = Image.open(requests.get(url, stream=True).raw)
    print(f"   size: {img.size}")
    images.append(img)
    placeholder += f"<|image_{i}|>\n"

messages = [
    {
        "role": "user",
        "content": placeholder
        + "Summarize the deck of slides and a long one to overcome some limitations.",
    },
]

prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

batch_size = 2
print(f"-- create inputs with batch_size={batch_size!r}")
if images:
    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")
    print(f"-- image_sizes {inputs.image_sizes}")
else:
    inputs = processor(prompt, return_tensors="pt").to("cuda:0")
    # To see what's happening if the batch size is changed.
    if batch_size > 1:
        inputs.data["input_ids"] = inputs.data["input_ids"].expand(
            (batch_size, inputs.data["input_ids"].shape[1])
        )
        inputs.data["attention_mask"] = inputs.data["attention_mask"].expand(
            (batch_size, inputs.data["attention_mask"].shape[1])
        )

print(f"-- types: {string_type(inputs, with_shape=True, with_min_max=True)}")

generation_args = {
    "max_new_tokens": 30,
    "temperature": 0.0,
    "do_sample": False,
}

inputs_iteration = []


def rewrite_forward(f, *args, **kwargs):
    print(f"------------- iteration {len(inputs_iteration)}")
    print(f"args: {string_type(args, with_shape=True, with_min_max=True)}")
    print(f"kwargs: {string_type(kwargs, with_shape=True, with_min_max=True)}")
    print(kwargs["input_ids"])
    inputs_iteration.append((args, kwargs))
    return f(*args, **kwargs)


print("-- intercept forward")
print(f"-- inputs type: {string_type(inputs)}")

model_forward = model.forward
model.forward = lambda f=model_forward, *args, **kwargs: rewrite_forward(f, *args, **kwargs)

generate_ids = model.generate(
    **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
)

# remove input tokens
generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("-- response")
# args: ()
# kwargs: dict(input_ids:T7s1x777[-1:32010],
#              position_ids:T7s1x777[0:776],
#              past_key_values:DynamicCache,
#                   #32[  T16s1x32x875x96[-8.375:6.5625], T16s1x32x875x96[-8.9375:8.875] ]
#              use_cache:int, True
#              attention_mask:T7s1x777, (boolean)
#              pixel_values:T1s1x5x3x336x336,
#              image_sizes:T7s1X2,
#              return_dict:int)
# kwargs: dict(input_ids:T7s1x777[-1:32010],position_ids:T7s1x777[0:776],
# past_key_values:DynamicCache(key_cache=[], DynamicCache(value_cache=[]),use_cache:int[True],
# attention_mask:T7s1x777[1:1],
# pixel_values:T1s1x5x3x336x336[-2.063474178314209:2.305359125137329],
# image_sizes:T7s1x2[672:672],return_dict:int[True])
print(response)


print("---------------------")
print(inputs_iteration)
