===========================
Long Outputs uneasy to read
===========================

examples
========

.. _l-long-outputs-llama-diff-export:

plot_llama_diff_export
++++++++++++++++++++++

**attention**

::

    1 ~ | INITIA float32  512x512         AXFK            onnx::MatMul_131                           | INITIA float32                  BAAA            ortshared_1_0_1_1_token_164               
    2 + |                                                                                            | INITIA int64    3               CKSA            ortshared_7_1_3_2_token_162                
    3 - | INITIA float32  512x512         WMJW            onnx::MatMul_132                           |                                                                                           
    4 - | INITIA float32  512x512         SCHH            onnx::MatMul_133                           |                                                                                           
    5 = | INITIA float32  512x512         VXZD            onnx::MatMul_169                           | INITIA float32  512x512         VXZD            _attention_o_proj_1_t_3                   
    6 + |                                                                                            | INITIA float32                  IAAA            ortshared_1_0_1_0_token_163                
    7 ~ | INITIA int64    4               CKIM            ortshared_7_1_4_0_token_76                 | INITIA int64    4               CIKM            ortshared_7_1_4_1_token_159               
    8 + |                                                                                            | INITIA int64    2               USAA            ortshared_7_1_2_1_token_167                
    9 ~ | INITIA int64    1               AAAA            ortshared_7_1_1_2_token_75                 | INITIA int64    4               CIKK            ortshared_7_1_4_0_token_154               
    10 ~ | INITIA int64    1               DAAA            ortshared_7_1_1_1_token_74                 | INITIA int64    4               CKIM            ortshared_7_1_4_2_token_165               
    11 + |                                                                                            | INITIA int64    3               QKMA            ortshared_7_1_3_1_token_158                
    12 = | INITIA float32  1024x64         CJYF            /attention/rotary_emb/Constant_output_0    | INITIA float32  1024x64         CJYF            _attention_1__val_22                      
    13 + |                                                                                            | INITIA int64                    ZAAA            ortshared_7_0_1_1_token_171                
    14 + |                                                                                            | INITIA int64                    BAAA            ortshared_7_0_1_0_token_156                
    15 = | INITIA float32  1024x64         GSEC            /attention/rotary_emb/Constant_1_output_0  | INITIA float32  1024x64         GSEC            _attention_1__val_32                      
    16 ~ | INITIA int64    1               SAAA            ortshared_7_1_1_0_token_73                 | INITIA int64    1               GAAA            ortshared_7_1_1_2_token_166               
    17 + |                                                                                            | INITIA float32  512x512         WMJW            _attention_k_proj_1_t_1                    
    18 + |                                                                                            | INITIA int64    1               AAAA            ortshared_7_1_1_0_token_155                
    19 + |                                                                                            | INITIA float32  512x512         SCHH            _attention_v_proj_1_t_2                    
    20 + |                                                                                            | INITIA float32  512x512         AXFK            _attention_q_proj_1_t                      
    21 + |                                                                                            | INITIA int64    1               DAAA            ortshared_7_1_1_4_token_170                
    22 = | INITIA int64    1               BAAA            ortshared_7_1_1_3_token_78                 | INITIA int64    1               BAAA            ortshared_7_1_1_3_token_169               
    23 + |                                                                                            | INITIA int64    1               SAAA            ortshared_7_1_1_1_token_160                
    24 ~ | INITIA int64    3               CKSA            ortshared_7_1_3_0_token_80                 | INITIA int64    3               QKKA            ortshared_7_1_3_0_token_157               
    25 + |                                                                                            | INITIA int64    3               QMKA            ortshared_7_1_3_3_token_168                
    26 ~ | INITIA int64    1               GAAA            ortshared_7_1_1_4_token_79                 | INITIA int64    2               BKAA            ortshared_7_1_2_0_token_161               
    27 = | INPUT  float32  2x1024x512      KKXP            input                                      | INPUT  float32  2x1024x512      KKXP            l_hidden_states_                          
    28 = | INPUT  float32  2x1x1024x1024   AAAA            onnx::Add_1                                | INPUT  float32  2x1x1024x1024   AAAA            l_attention_mask_                         
    29 = | INPUT  int64    1x1024          KAQG            position_ids                               | INPUT  int64    1x1024          KAQG            l_position_ids_                           
    30 + |                                                                                            | RESULT float32  2048x512        KKXP Reshape    _attention_v_proj_1_view_4                 
    31 + |                                                                                            | RESULT float32  2048x512        DTNI MatMul     _attention_v_proj_1_mm_2                   
    32 + |                                                                                            | RESULT float32  2x1024x512      DTNI Reshape    _attention_1_attention_v_proj_1            
    33 + |                                                                                            | RESULT float32  2x1024x8x64     DTNI Reshape    _attention_1_view_8                        
    34 + |                                                                                            | RESULT float32  2x8x1024x64     BVMJ Transpose  _attention_1_transpose_2                   
    35 + |                                                                                            | RESULT float32  16x1024x64      BVMJ Reshape    _attention_1_view_13                       
    36 + |                                                                                            | RESULT float32  2x1x1024x1024   AAAA Mul        _inlfunc_aten_add|folded_2_other_1         
    37 + |                                                                                            | RESULT int64    1x1024          KAQG Expand     _attention_1__val_35                       
    38 + |                                                                                            | RESULT int64    1x1024x1        KAQG Unsqueeze  _attention_1__val_37                       
    39 + |                                                                                            | RESULT int64    1x1024x1        KAQG Concat     _attention_1__val_38                       
    40 ~ | RESULT float32  1x1024x64       GSEC Gather     /attention/Gather_1_output_0               | RESULT float32  1x1024x64       GSEC GatherND   _attention_1__val_39                      
    41 = | RESULT float32  1x1x1024x64     GSEC Unsqueeze  /attention/Unsqueeze_1_output_0            | RESULT float32  1x1x1024x64     GSEC Unsqueeze  _attention_1_aten_unsqueeze_65_n2         
    42 = | RESULT float32  1x1024x1x64     GSEC Transpose  Transpose_token_4_out0                     | RESULT float32  1x1024x1x64     GSEC Transpose  Transpose_token_5_out0                    
    43 + |                                                                                            | RESULT float32  2048x512        RCRG MatMul     _attention_k_proj_1_mm_1                   
    44 ~ | RESULT float32  2x1024x512      RCRG MatMul     /attention/k_proj/MatMul_output_0          | RESULT float32  2x1024x512      RCRG Reshape    _attention_1_attention_k_proj_1           
    45 = | RESULT float32  2x1024x8x64     RCRG Reshape    /attention/Reshape_1_output_0              | RESULT float32  2x1024x8x64     RCRG Reshape    _attention_1_view_7                       
    46 = | RESULT float32  2x1024x8x32     DJVL Slice      /attention/Slice_3                         | RESULT float32  2x1024x8x32     DJVL Slice      _attention_1_Slice_140                    
    47 = | RESULT float32  2x1024x8x32     XRFP Neg        /attention/Neg_1                           | RESULT float32  2x1024x8x32     XRFP Neg        _attention_1_aten_neg_141_n0              
    48 = | RESULT float32  2x1024x8x32     OUWV Slice      /attention/Slice_2                         | RESULT float32  2x1024x8x32     OUWV Slice      _attention_1_Slice_123                    
    49 = | RESULT float32  2x1024x8x64     LLBK Concat     /attention/Concat_1                        | RESULT float32  2x1024x8x64     LLBK Concat     _attention_1_aten_cat_143_n0              
    50 = | RESULT float32  2x1024x8x64     AULV Mul        /attention/Mul_3                           | RESULT float32  2x1024x8x64     AULV Mul        _attention_1_aten_mul_144_n0              
    51 ~ | RESULT float32  1x1024x64       CJYF Gather     /attention/Gather_output_0                 | RESULT float32  1x1024x64       CJYF GatherND   _attention_1__val_29                      
    52 = | RESULT float32  1x1x1024x64     CJYF Unsqueeze  /attention/Unsqueeze_output_0              | RESULT float32  1x1x1024x64     CJYF Unsqueeze  _attention_1_aten_unsqueeze_55_n2         
    53 = | RESULT float32  1x1024x1x64     CJYF Transpose  Transpose_token_6_out0                     | RESULT float32  1x1024x1x64     CJYF Transpose  Transpose_token_8_out0                    
    54 = | RESULT float32  2x1024x8x64     IAJA Mul        /attention/Mul_2                           | RESULT float32  2x1024x8x64     IAJA Mul        _attention_1_aten_mul_106_n0              
    55 = | RESULT float32  2x1024x8x64     JTVV Add        /attention/Add_1                           | RESULT float32  2x1024x8x64     JTVV Add        _inlfunc_aten_add|folded_1_n3             
    56 = | RESULT float32  2x8x64x1024     NQOB Transpose  /attention/Transpose_3_output_0            | RESULT float32  2x8x64x1024     NQOB Transpose  _attention_1_transpose_3                  
    57 + |                                                                                            | RESULT float32  16x64x1024      NQOB Reshape    _attention_1_view_10                       
    58 + |                                                                                            | RESULT float32  1x1x1024x64     GSEC Transpose  _attention_1_unsqueeze_1                   
    59 + |                                                                                            | RESULT float32  2048x512        YNJI MatMul     _attention_q_proj_1_mm                     
    60 ~ | RESULT float32  2x1024x512      YNJI MatMul     /attention/q_proj/MatMul_output_0          | RESULT float32  2x1024x512      YNJI Reshape    _attention_1_attention_q_proj_1           
    61 = | RESULT float32  2x1024x8x64     YNJI Reshape    /attention/Reshape_output_0                | RESULT float32  2x1024x8x64     YNJI Reshape    _attention_1_view_6                       
    62 = | RESULT float32  2x8x1024x64     MABQ Transpose  /attention/Transpose_output_0              | RESULT float32  2x8x1024x64     MABQ Transpose  _attention_1_transpose                    
    63 = | RESULT float32  2x8x1024x32     ERKV Slice      /attention/Slice_1_output_0                | RESULT float32  2x8x1024x32     ERKV Slice      _attention_1_slice_4                      
    64 = | RESULT float32  2x8x1024x32     WJQF Neg        /attention/Neg_output_0                    | RESULT float32  2x8x1024x32     WJQF Neg        _attention_1_neg                          
    65 = | RESULT float32  2x8x1024x32     HKSV Slice      /attention/Slice_output_0                  | RESULT float32  2x8x1024x32     HKSV Slice      _attention_1_slice_3                      
    66 = | RESULT float32  2x8x1024x64     DSIZ Concat     /attention/Concat_output_0                 | RESULT float32  2x8x1024x64     DSIZ Concat     _attention_1_cat                          
    67 = | RESULT float32  2x8x1024x64     NTZT Mul        /attention/Mul_1_output_0                  | RESULT float32  2x8x1024x64     NTZT Mul        _attention_1_mul_1                        
    68 + |                                                                                            | RESULT float32  1x1x1024x64     CJYF Transpose  _attention_1_unsqueeze                     
    69 = | RESULT float32  2x8x1024x64     SDFX Mul        /attention/Mul_output_0                    | RESULT float32  2x8x1024x64     SDFX Mul        _attention_1_mul                          
    70 = | RESULT float32  2x8x1024x64     GWEQ Add        /attention/Add_output_0                    | RESULT float32  2x8x1024x64     GWEQ Add        _attention_1_add                          
    71 + |                                                                                            | RESULT float32  16x1024x64      GWEQ Reshape    _attention_1_view_9                        
    72 + |                                                                                            | RESULT float32  16x1024x1024    ISCK MatMul     _attention_1_bmm                           
    73 + |                                                                                            | RESULT float32  2x8x1024x1024   ISCK Reshape    _attention_1_view_11                       
    74 ~ | RESULT float32  2x8x1024x1024   YSTO FusedMatMu /attention/Div_output_0                    | RESULT float32  2x8x1024x1024   YSTO Div        _attention_1_div                          
    75 = | RESULT float32  2x8x1024x1024   YSTO Add        /attention/Add_2_output_0                  | RESULT float32  2x8x1024x1024   YSTO Add        _attention_1_add_2                        
    76 ~ | RESULT float32  2x8x1024x1024   NONO Softmax    /attention/Softmax_output_0                | RESULT float32  2x8x1024x1024   NNNO Softmax    _attention_1_aten_softmax_no_dtype_163_res
    77 + |                                                                                            | RESULT float32  16x1024x1024    NNNO Reshape    _attention_1_view_12                       
    78 - | RESULT float32  2x1024x512      DTNI MatMul     /attention/v_proj/MatMul_output_0          |                                                                                           
    79 ~ | RESULT float32  2x1024x8x64     DTNI Reshape    /attention/Reshape_2_output_0              | RESULT float32  16x1024x64      BUPD MatMul     _attention_1_bmm_1                        
    80 ~ | RESULT float32  2x8x1024x64     BVMJ Transpose  /attention/Transpose_2_output_0            | RESULT float32  2x8x1024x64     BUPD Reshape    _attention_1_view_14                      
    81 + |                                                                                            | RESULT float32  2x1024x8x64     NITB Transpose  _attention_1_transpose_4                   
    82 ~ | RESULT float32  2x8x1024x64     BUPD MatMul     /attention/MatMul_1_output_0               | RESULT float32  2x1024x512      NITB Reshape    _attention_1_view_15                      
    83 ~ | RESULT float32  2x1024x8x64     NITB Transpose  /attention/Transpose_4_output_0            | RESULT float32  2048x512        NITB Reshape    _attention_o_proj_1_view_16               
    84 ~ | RESULT float32  2x1024x512      NITB Reshape    /attention/Reshape_3_output_0              | RESULT float32  2048x512        XTSR MatMul     _attention_o_proj_1_mm_3                  
    85 ~ | RESULT float32  2x1024x512      XTSR MatMul     130                                        | RESULT float32  2x1024x512      XTSR Reshape    attention_1                               
    86 = | OUTPUT float32  2x1024x512      XTSR            130                                        | OUTPUT float32  2x1024x512      XTSR            attention_1             