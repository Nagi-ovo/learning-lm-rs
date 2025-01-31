use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use num_traits::Num;
use safetensors::SafeTensors;

pub struct LLamaParams<T: Num> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

trait GetTensorFromSafeTensors<P: Num> {
    fn get_tensor_from(tensors: &SafeTensors, name: &str) -> Result<Tensor<P>, &'static str>;
}

impl GetTensorFromSafeTensors<f32> for f32 {
    fn get_tensor_from(tensors: &SafeTensors, name: &str) -> Result<Tensor<f32>, &'static str> {
        let tensor_view = tensors.tensor(name).map_err(|e| {
            assert!(matches!(e, safetensors::SafeTensorError::TensorNotFound(_)));
            "Tensor not found"
        })?;

        let data = match tensor_view.dtype() {
            safetensors::Dtype::F32 => tensor_view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            _ => return Err("Unsupported data type"),
        };

        Ok(Tensor::new(data, &tensor_view.shape().to_vec()))
    }
}

macro_rules! get_tensor_vec {
    ($tensors:expr, $pattern:literal, $layers:expr) => {{
        (0..$layers)
            .map(|i| {
                let name = format!($pattern, i);
                f32::get_tensor_from($tensors, &name).unwrap()
            })
            .collect::<Vec<_>>()
    }};
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let embedding_table = if config.tie_word_embeddings {
            f32::get_tensor_from(safetensor, "lm_head.weight").unwrap()
        } else {
            f32::get_tensor_from(safetensor, "model.embed_tokens.weight").unwrap()
        };

        let n_layers = config.num_hidden_layers;

        Self {
            embedding_table,
            rms_att_w: get_tensor_vec!(
                safetensor,
                "model.layers.{}.input_layernorm.weight",
                n_layers
            ),
            wq: get_tensor_vec!(
                safetensor,
                "model.layers.{}.self_attn.q_proj.weight",
                n_layers
            ),
            wk: get_tensor_vec!(
                safetensor,
                "model.layers.{}.self_attn.k_proj.weight",
                n_layers
            ),
            wv: get_tensor_vec!(
                safetensor,
                "model.layers.{}.self_attn.v_proj.weight",
                n_layers
            ),
            wo: get_tensor_vec!(
                safetensor,
                "model.layers.{}.self_attn.o_proj.weight",
                n_layers
            ),
            rms_ffn_w: get_tensor_vec!(
                safetensor,
                "model.layers.{}.post_attention_layernorm.weight",
                n_layers
            ),
            w_up: get_tensor_vec!(safetensor, "model.layers.{}.mlp.up_proj.weight", n_layers),
            w_gate: get_tensor_vec!(safetensor, "model.layers.{}.mlp.gate_proj.weight", n_layers),
            w_down: get_tensor_vec!(safetensor, "model.layers.{}.mlp.down_proj.weight", n_layers),
            rms_out_w: f32::get_tensor_from(safetensor, "model.norm.weight").unwrap(),
            lm_head: f32::get_tensor_from(safetensor, "lm_head.weight").unwrap(),
        }
    }
}
