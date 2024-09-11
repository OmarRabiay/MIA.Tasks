
# TASK12.5 - Clean Encoder (BONUS)

### About

A differential robot with DC motors is equipped with rotary encoders that have 7 pulses per revolution (PPR). Each motor has a 30:1 gearbox and is paired with a wheel that has a diameter of 20 cm. The maximum speed of the robot is 1 m/s. Given that the encoder signal is noisy, apply a practical first-order low-pass filter (LPF) with a suitable cutoff frequency to mitigate the noise.


### 1. Motor Specifications:
- **Pulses per revolution (PPR):** 7
- **Gearbox ratio:** 30:1
- **Wheel diameter:** 20 cm (radius = 10 cm)
- **Max robot speed:** 1 m/s

### 2. Max Wheel RPM Calculation:
The relationship between wheel speed, robot speed, and RPM is:
``
v = ω * r
``

where:
- \( v \) = 1 m/s (max speed)
- \( r \) = 0.1 m (wheel radius)
- \( ω \) is the angular velocity  in rad/s.

First, solve for \( ω \):

``ω = v / r = 1 / 0.1 = 10 rad/s``



Convert rad/s to revolutions per minute (RPM):

``RPM = (ω * 60) / (2π) = (10 * 60) / (2π) ≈ 95.49 RPM``



### 3. Max Motor Shaft RPM:
Since the motor has a 30:1 gearbox, the motor shaft RPM is higher:

``Motor Shaft RPM = 95.49 * 30 = 2864.7 RPM``


### 4. Encoder Pulse Frequency:
The number of encoder pulses per motor revolution is 7 (PPR). Thus, the max encoder pulse frequency is:

``Pulse Frequency = 2864.7 * 7 = 20052.9 pulses per minute``

Convert this to pulses per second (Hz):

``Pulse Frequency = 20052.9 / 60 ≈ 334.2 Hz``



### 5. Noise Mitigation with Low-Pass Filter:
Since the encoder signal is noisy, a first-order LPF can be designed with a cutoff frequency slightly higher than the max signal frequency to attenuate high-frequency noise while preserving the actual signal.

A common choice for the cutoff frequency ( fc ) is between 1.5x and 2x the max signal frequency to avoid too much attenuation of the encoder pulses. Based on the max frequency of 334.2 Hz, we can choose a cutoff frequency of around **500Hz - 600Hz**.

### 6. First-Order LPF Formula:
The transfer function for a first-order low-pass filter is:

``H(s) =  ωc / (s + ωc)``



where \( ωc \) is the cutoff angular frequency. Using a chosen cutoff frequency of 500 Hz:

``ωc = 2π * 500 = 3141.6 rad/s``



### 7. Implementing the Filter:
You can implement this LPF in software using the following discrete-time approximation (for a real-time system):

``y[n] = α x[n] + (1 - α) y[n-1]``



where:

``  α = Δt / (Δt + 1 / (2π fc))``

\( Δt \) is the sampling period, and \( fc \) is the cutoff frequency.

## Final Example Code
``` 
import numpy as np

# 1. Generate a Noisy Signal

sampling_freq = 1000  
duration = 1  
signal_freq = 334.2  
noise_amplitude = 0.5  

t = np.linspace(0, duration, int(sampling_freq * duration), endpoint=False)

signal = np.sin(2 * np.pi * signal_freq * t)

noise = noise_amplitude * np.random.randn(len(t))
noisy_signal = signal + noise


# 2. Design the Low-Pass Filter

cutoff_freq = 500
dt = 1.0 / sampling_freq

w0 = 2 * np.pi * cutoff_freq
alpha = dt / (dt + (1 / w0))

filtered_signal = np.zeros_like(noisy_signal)

# 3. Apply the Filter

for i in range(1, len(noisy_signal)):
    filtered_signal[i] = alpha * noisy_signal[i] + (1 - alpha) * filtered_signal[i-1]

```
