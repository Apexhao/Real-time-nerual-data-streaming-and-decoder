"""
A3D3 Project: Real-time Neural Decoder for Brain-Computer Interfaces
Real-time Neural Decoder for Brain-Computer Interfaces

This module implements a real-time neural decoder using a pre-trained LFADS model.
It supports continuous decoding of neural signals with the ability to dynamically
reset hidden states during experiments.

Author: Hao Fang
Date: 10/8/2014
Copyright: 2024, The Trustees of University of Washington
"""

import tensorflow as tf
import numpy as np
from collections import deque
import threading
import sys
import select
import tty
import termios
from typing import Generator, List, Tuple, Optional
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeDecoder:
    """
    A class for real-time decoding of neural signals using a sliding window approach.
    """

    def __init__(self, model: tf.keras.Model, window_size: int, step_size: int):
        """
        Initialize the RealTimeDecoder.

        Args:
            model (tf.keras.Model): The pre-trained LFADS model.
            window_size (int): The number of time steps in each decoding window.
            step_size (int): The number of time steps to slide the window.
        """
        self.model = model
        self.window_size = window_size
        self.step_size = step_size
        self.buffer = deque(maxlen=window_size)
        self.hidden_states = deque(maxlen=window_size)
        self.initial_state = None
        self.reset_flag = False

    @tf.function
    def decode_window(self, window: tf.Tensor, initial_state: Optional[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Decode a single window of neural data.

        Args:
            window (tf.Tensor): The input window of neural data.
            initial_state (tf.Tensor): The initial hidden state for the window.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The decoded output and all intermediate states.
        """
        window = tf.expand_dims(window, axis=0)
        decoded_output, all_states = self.model(window, initial_state=initial_state, 
                                                return_state_sequence=True, training=False)
        return decoded_output[0], all_states

    def reset_hidden_states(self) -> None:
        """Reset the hidden states and initial state."""
        self.hidden_states.clear()
        self.initial_state = None
        self.reset_flag = False
        logger.info("Hidden states have been reset.")

    def process_sample(self, sample: tf.Tensor) -> Optional[tf.Tensor]:
        """
        Process a single sample of neural data.

        Args:
            sample (tf.Tensor): A single time step of neural data.

        Returns:
            Optional[tf.Tensor]: Decoded output if available, None otherwise.
        """
        if self.reset_flag:
            self.reset_hidden_states()
        
        self.buffer.append(sample)
        
        if len(self.buffer) == self.window_size:
            window = tf.stack(list(self.buffer))
            decoded, all_states = self.decode_window(window, self.initial_state)
            
            self.hidden_states.extend(all_states)
            self.initial_state = self.hidden_states[self.step_size - 1]
            
            for _ in range(self.step_size):
                self.buffer.popleft()
                if self.hidden_states:
                    self.hidden_states.popleft()
            
            return decoded[-self.step_size:]
        
        return None

def keyboard_input(decoder: RealTimeDecoder) -> None:
    """
    Handle keyboard input for real-time control.

    Args:
        decoder (RealTimeDecoder): The decoder instance to control.
    """
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key.lower() == 'r':
                    decoder.reset_flag = True
                    logger.info("Reset requested.")
                elif key.lower() == 'q':
                    logger.info("Quitting...")
                    break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def simulate_real_time_processing(data_stream: Generator[tf.Tensor, None, None], 
                                  num_samples: int, 
                                  decoder: RealTimeDecoder) -> tf.Tensor:
    """
    Simulate real-time processing of neural data.

    Args:
        data_stream (Generator): A generator yielding neural data samples.
        num_samples (int): The number of samples to process.
        decoder (RealTimeDecoder): The decoder instance to use.

    Returns:
        tf.Tensor: The concatenated decoded outputs.
    """
    decoded_outputs = []
    
    input_thread = threading.Thread(target=keyboard_input, args=(decoder,))
    input_thread.daemon = True
    input_thread.start()
    
    try:
        for _ in range(num_samples):
            new_sample = next(data_stream)
            decoded = decoder.process_sample(new_sample)
            
            if decoded is not None:
                decoded_outputs.append(decoded)
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    
    return tf.concat(decoded_outputs, axis=0) if decoded_outputs else tf.constant([])

def load_lfads_model(model_path: str) -> tf.keras.Model:
    """
    Load a pre-trained LFADS model.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        tf.keras.Model: The loaded LFADS model.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"LFADS model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the real-time neural decoder.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    model = load_lfads_model(args.model_path)
    decoder = RealTimeDecoder(model, args.window_size, args.step_size)

    def data_stream_generator() -> Generator[tf.Tensor, None, None]:
        while True:
            yield tf.random.normal((args.num_features,))

    data_stream = data_stream_generator()

    logger.info("Starting real-time decoding. Press 'R' to reset hidden states, 'Q' to quit.")
    decoded_signal = simulate_real_time_processing(data_stream, args.num_samples, decoder)
    logger.info(f"Decoded signal shape: {decoded_signal.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Neural Decoder for BCI")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained LFADS model")
    parser.add_argument("--window_size", type=int, default=100, help="Number of time steps in each decoding window")
    parser.add_argument("--step_size", type=int, default=10, help="Number of time steps to slide the window")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to process")
    parser.add_argument("--num_features", type=int, default=100, help="Number of features in each neural data sample")
    
    args = parser.parse_args()
    
    main(args)
