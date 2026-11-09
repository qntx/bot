#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut lines = ovo_ai::http::sse::LineDecoder::default();
    let mut decoder = ovo_ai::http::sse::SseDecoder::default();
    // Split at every byte — worst-case chunking.
    for byte in data.chunks(1).chain([b""].iter().copied()) {
        for line in lines.decode(byte) {
            let _ = decoder.decode(&line);
        }
    }
    for line in lines.flush() {
        let _ = decoder.decode(&line);
    }
    let _ = decoder.flush();
});
