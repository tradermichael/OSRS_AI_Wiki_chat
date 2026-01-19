class PTTCaptureProcessor extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const input = inputs && inputs[0] && inputs[0][0] ? inputs[0][0] : null;
    const output = outputs && outputs[0] && outputs[0][0] ? outputs[0][0] : null;

    if (output && input) {
      // Pass through (will be muted by a 0-gain node on the main thread).
      output.set(input);
    } else if (output) {
      output.fill(0);
    }

    if (input) {
      // Post the raw Float32 PCM up to main thread.
      // Structured clone is fine here; main thread downsamples + encodes.
      this.port.postMessage(input);
    }
    return true;
  }
}

registerProcessor('ptt-capture', PTTCaptureProcessor);
