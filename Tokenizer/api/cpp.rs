use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr, CString};
use std::os::raw::{c_int, c_uint};
use std::ptr;
use std::slice;

use rustc_hash::FxHashMap as HashMap;

use crate::{byte_pair_encode, CoreBPE, Rank, DecodeKeyError};

// Opaque struct to hold CoreBPE for C++ use
#[repr(C)]
pub struct CoreBPEHandle(*mut CoreBPE);

// Result struct for encoded tokens
#[repr(C)]
pub struct EncodedTokens {
    tokens: *mut Rank,
    length: usize,
}

// Result struct for decoded bytes
#[repr(C)]
pub struct DecodedBytes {
    bytes: *mut u8,
    length: usize,
}

// Error codes
#[repr(C)]
pub enum TokenizerError {
    Success = 0,
    InvalidToken = 1,
    DecodingError = 2,
    EncodingError = 3,
    InvalidArgument = 4,
    OutOfMemory = 5,
}

// Function to safely convert C strings to Rust strings
unsafe fn c_str_to_string(ptr: *const c_char) -> Result<String, TokenizerError> {
    if ptr.is_null() {
        return Err(TokenizerError::InvalidArgument);
    }
    
    match CStr::from_ptr(ptr).to_str() {
        Ok(s) => Ok(s.to_owned()),
        Err(_) => Err(TokenizerError::InvalidArgument),
    }
}

// Create a new CoreBPE instance
#[no_mangle]
pub extern "C" fn create_core_bpe(
    encoder_bytes: *const *const u8,
    encoder_sizes: *const usize,
    encoder_ranks: *const Rank,
    encoder_count: usize,
    special_tokens_strs: *const *const c_char,
    special_tokens_ranks: *const Rank,
    special_tokens_count: usize,
    pattern: *const c_char,
    err_code: *mut TokenizerError,
) -> CoreBPEHandle {
    // Default error code
    if !err_code.is_null() {
        unsafe { *err_code = TokenizerError::Success; }
    }

    // Validate pattern
    let pattern_str = match unsafe { c_str_to_string(pattern) } {
        Ok(s) => s,
        Err(e) => {
            if !err_code.is_null() {
                unsafe { *err_code = e; }
            }
            return CoreBPEHandle(ptr::null_mut());
        }
    };

    // Create encoder HashMap
    let mut encoder = HashMap::default();
    for i in 0..encoder_count {
        unsafe {
            let token_bytes = std::slice::from_raw_parts(
                *encoder_bytes.add(i), 
                *encoder_sizes.add(i)
            );
            let rank = *encoder_ranks.add(i);
            encoder.insert(token_bytes.to_vec(), rank);
        }
    }

    // Create special tokens HashMap
    let mut special_tokens_encoder = HashMap::default();
    for i in 0..special_tokens_count {
        unsafe {
            if let Ok(token_str) = c_str_to_string(*special_tokens_strs.add(i)) {
                let rank = *special_tokens_ranks.add(i);
                special_tokens_encoder.insert(token_str, rank);
            } else {
                if !err_code.is_null() {
                    *err_code = TokenizerError::InvalidArgument;
                }
                return CoreBPEHandle(ptr::null_mut());
            }
        }
    }

    // Create CoreBPE instance
    match CoreBPE::new_internal(encoder, special_tokens_encoder, &pattern_str) {
        Ok(core_bpe) => {
            let boxed = Box::new(core_bpe);
            CoreBPEHandle(Box::into_raw(boxed))
        }
        Err(_) => {
            if !err_code.is_null() {
                unsafe { *err_code = TokenizerError::EncodingError; }
            }
            CoreBPEHandle(ptr::null_mut())
        }
    }
}

// Free the CoreBPE instance
#[no_mangle]
pub extern "C" fn free_core_bpe(handle: CoreBPEHandle) {
    if !handle.0.is_null() {
        unsafe {
            drop(Box::from_raw(handle.0));
        }
    }
}

// Encode ordinary text
#[no_mangle]
pub extern "C" fn encode_ordinary(
    handle: CoreBPEHandle,
    text: *const c_char,
    err_code: *mut TokenizerError,
) -> EncodedTokens {
    // Default to empty result
    let empty_result = EncodedTokens {
        tokens: ptr::null_mut(),
        length: 0,
    };

    // Check if handle is valid
    if handle.0.is_null() {
        if !err_code.is_null() {
            unsafe { *err_code = TokenizerError::InvalidArgument; }
        }
        return empty_result;
    }

    // Convert C string to Rust string
    let text_str = match unsafe { c_str_to_string(text) } {
        Ok(s) => s,
        Err(e) => {
            if !err_code.is_null() {
                unsafe { *err_code = e; }
            }
            return empty_result;
        }
    };

    // Call CoreBPE method
    let core_bpe = unsafe { &*handle.0 };
    let tokens = core_bpe.encode_ordinary(&text_str);

    // Allocate memory for result
    if tokens.is_empty() {
        if !err_code.is_null() {
            unsafe { *err_code = TokenizerError::Success; }
        }
        return empty_result;
    }

    let tokens_len = tokens.len();
    let tokens_ptr = unsafe {
        let layout = std::alloc::Layout::array::<Rank>(tokens_len).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut Rank;
        if ptr.is_null() {
            if !err_code.is_null() {
                *err_code = TokenizerError::OutOfMemory;
            }
            return empty_result;
        }

        // Copy tokens to allocated memory
        ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens_len);
        ptr
    };

    if !err_code.is_null() {
        unsafe { *err_code = TokenizerError::Success; }
    }
    
    EncodedTokens {
        tokens: tokens_ptr,
        length: tokens_len,
    }
}

// Encode with special tokens
#[no_mangle]
pub extern "C" fn encode(
    handle: CoreBPEHandle,
    text: *const c_char,
    allowed_special: *const *const c_char,
    allowed_special_count: usize,
    err_code: *mut TokenizerError,
) -> EncodedTokens {
    // Default to empty result
    let empty_result = EncodedTokens {
        tokens: ptr::null_mut(),
        length: 0,
    };

    // Check if handle is valid
    if handle.0.is_null() {
        if !err_code.is_null() {
            unsafe { *err_code = TokenizerError::InvalidArgument; }
        }
        return empty_result;
    }

    // Convert C string to Rust string
    let text_str = match unsafe { c_str_to_string(text) } {
        Ok(s) => s,
        Err(e) => {
            if !err_code.is_null() {
                unsafe { *err_code = e; }
            }
            return empty_result;
        }
    };

    // Build allowed special tokens HashSet
    let mut allowed_special_set = HashSet::new();
    for i in 0..allowed_special_count {
        unsafe {
            if let Ok(special_token) = c_str_to_string(*allowed_special.add(i)) {
                allowed_special_set.insert(special_token);
            } else {
                if !err_code.is_null() {
                    *err_code = TokenizerError::InvalidArgument;
                }
                return empty_result;
            }
        }
    }

    // Call CoreBPE method
    let core_bpe = unsafe { &*handle.0 };
    let allowed_special_refs: HashSet<&str> = allowed_special_set.iter().map(|s| s.as_str()).collect();
    let (tokens, _) = core_bpe.encode(&text_str, &allowed_special_refs);

    // Allocate memory for result
    if tokens.is_empty() {
        if !err_code.is_null() {
            unsafe { *err_code = TokenizerError::Success; }
        }
        return empty_result;
    }

    let tokens_len = tokens.len();
    let tokens_ptr = unsafe {
        let layout = std::alloc::Layout::array::<Rank>(tokens_len).unwrap();
        let ptr = std::alloc::alloc(layout) as *mut Rank;
        if ptr.is_null() {
            if !err_code.is_null() {
                *err_code = TokenizerError::OutOfMemory;
            }
            return empty_result;
        }

        // Copy tokens to allocated memory
        ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens_len);
        ptr
    };

    if !err_code.is_null() {
        unsafe { *err_code = TokenizerError::Success; }
    }
    
    EncodedTokens {
        tokens: tokens_ptr,
        length: tokens_len,
    }
}

// Decode tokens to bytes
#[no_mangle]
pub extern "C" fn decode_bytes(
    handle: CoreBPEHandle,
    tokens: *const Rank,
    tokens_length: usize,
    err_code: *mut TokenizerError,
) -> DecodedBytes {
    // Default to empty result
    let empty_result = DecodedBytes {
        bytes: ptr::null_mut(),
        length: 0,
    };

    // Check if handle is valid
    if handle.0.is_null() {
        if !err_code.is_null() {
            unsafe { *err_code = TokenizerError::InvalidArgument; }
        }
        return empty_result;
    }

    // Get tokens slice
    let tokens_slice = unsafe { slice::from_raw_parts(tokens, tokens_length) };

    // Call CoreBPE method
    let core_bpe = unsafe { &*handle.0 };
    let bytes_result = core_bpe.decode_bytes(tokens_slice);

    match bytes_result {
        Ok(bytes) => {
            if bytes.is_empty() {
                if !err_code.is_null() {
                    unsafe { *err_code = TokenizerError::Success; }
                }
                return empty_result;
            }

            let bytes_len = bytes.len();
            let bytes_ptr = unsafe {
                let layout = std::alloc::Layout::array::<u8>(bytes_len).unwrap();
                let ptr = std::alloc::alloc(layout) as *mut u8;
                if ptr.is_null() {
                    if !err_code.is_null() {
                        *err_code = TokenizerError::OutOfMemory;
                    }
                    return empty_result;
                }

                // Copy bytes to allocated memory
                ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes_len);
                ptr
            };

            if !err_code.is_null() {
                unsafe { *err_code = TokenizerError::Success; }
            }
            
            DecodedBytes {
                bytes: bytes_ptr,
                length: bytes_len,
            }
        },
        Err(_) => {
            if !err_code.is_null() {
                unsafe { *err_code = TokenizerError::InvalidToken; }
            }
            empty_result
        }
    }
}

// Encode a single token
#[no_mangle]
pub extern "C" fn encode_single_token(
    handle: CoreBPEHandle,
    bytes: *const u8,
    bytes_length: usize,
    out_token: *mut Rank,
    err_code: *mut TokenizerError,
) -> c_int {
    // Check if handle is valid
    if handle.0.is_null() || out_token.is_null() {
        if !err_code.is_null() {
            unsafe { *err_code = TokenizerError::InvalidArgument; }
        }
        return 0;
    }

    // Get bytes slice
    let bytes_slice = unsafe { slice::from_raw_parts(bytes, bytes_length) };

    // Call CoreBPE method
    let core_bpe = unsafe { &*handle.0 };
    
    // Try in encoder first
    if let Some(token) = core_bpe.encoder.get(bytes_slice) {
        unsafe {
            *out_token = *token;
            if !err_code.is_null() {
                *err_code = TokenizerError::Success;
            }
        }
        return 1;
    }
    
    // Then try in special tokens
    if let Ok(bytes_str) = std::str::from_utf8(bytes_slice) {
        if let Some(token) = core_bpe.special_tokens_encoder.get(bytes_str) {
            unsafe {
                *out_token = *token;
                if !err_code.is_null() {
                    *err_code = TokenizerError::Success;
                }
            }
            return 1;
        }
    }

    if !err_code.is_null() {
        unsafe { *err_code = TokenizerError::InvalidToken; }
    }
    0
}

// Free encoded tokens memory
#[no_mangle]
pub extern "C" fn free_tokens(tokens: EncodedTokens) {
    if !tokens.tokens.is_null() && tokens.length > 0 {
        unsafe {
            let layout = std::alloc::Layout::array::<Rank>(tokens.length).unwrap();
            std::alloc::dealloc(tokens.tokens as *mut u8, layout);
        }
    }
}

// Free decoded bytes memory
#[no_mangle]
pub extern "C" fn free_bytes(bytes: DecodedBytes) {
    if !bytes.bytes.is_null() && bytes.length > 0 {
        unsafe {
            let layout = std::alloc::Layout::array::<u8>(bytes.length).unwrap();
            std::alloc::dealloc(bytes.bytes, layout);
        }
    }
}

// Get the special tokens
#[no_mangle]
pub extern "C" fn get_special_tokens_count(handle: CoreBPEHandle) -> usize {
    if handle.0.is_null() {
        return 0;
    }
    
    let core_bpe = unsafe { &*handle.0 };
    core_bpe.special_tokens_encoder.len()
}

// Get special token key at index
#[no_mangle]
pub extern "C" fn get_special_token_at_index(
    handle: CoreBPEHandle,
    index: usize,
    out_token: *mut Rank,
    err_code: *mut TokenizerError,
) -> *mut c_char {
    if handle.0.is_null() || out_token.is_null() {
        if !err_code.is_null() {
            unsafe { *err_code = TokenizerError::InvalidArgument; }
        }
        return ptr::null_mut();
    }
    
    let core_bpe = unsafe { &*handle.0 };
    
    if index >= core_bpe.special_tokens_encoder.len() {
        if !err_code.is_null() {
            unsafe { *err_code = TokenizerError::InvalidArgument; }
        }
        return ptr::null_mut();
    }
    
    // Get the key and value at the specified index
    // Find the nth item in the HashMap
    let mut iter = core_bpe.special_tokens_encoder.iter();
    let (token_str, rank) = match iter.nth(index) {
        Some((k, v)) => (k, v),
        None => {
            if !err_code.is_null() {
                unsafe { *err_code = TokenizerError::InvalidArgument; }
            }
            return ptr::null_mut();
        }
    };
    
    // Set the token value
    unsafe { *out_token = *rank; }
    
    // Return the token string
    match CString::new(token_str.as_str()) {
        Ok(c_str) => {
            if !err_code.is_null() {
                unsafe { *err_code = TokenizerError::Success; }
            }
            c_str.into_raw()
        }
        Err(_) => {
            if !err_code.is_null() {
                unsafe { *err_code = TokenizerError::EncodingError; }
            }
            ptr::null_mut()
        }
    }
}

// Free a C string allocated by Rust
#[no_mangle]
pub extern "C" fn free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
} 