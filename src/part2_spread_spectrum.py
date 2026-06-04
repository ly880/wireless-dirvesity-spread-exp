"""Part 2: direct-sequence spread spectrum experiment."""

import numpy as np

from utils import (
    add_awgn,
    add_narrowband_interference,
    bpsk_demodulate,
    bpsk_modulate,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
    plot_correlation_snapshot,
)


def _validate_pn_chips(pn_chips):
    pn_chips = np.asarray(pn_chips, dtype=float)
    if pn_chips.ndim != 1 or len(pn_chips) == 0:
        raise ValueError('pn_chips must be a non-empty one-dimensional array')
    if not np.all(np.isin(pn_chips, [-1, 1])):
        raise ValueError('pn_chips must contain only +1 and -1')
    return pn_chips


def generate_m_sequence(register_state, taps, length=None):
    """
    Generate a bipolar m-sequence with an LFSR.

    Convention:
        register_state is listed from left to right.
        taps are 1-based positions from left to right.
        each clock outputs the rightmost bit, shifts right, and inserts feedback
        at the left. The feedback bit is XOR of tapped bits.

    Returns:
        chips in bipolar form: bit 0 -> +1, bit 1 -> -1.
    """
    state = np.asarray(register_state, dtype=int)
    taps = list(taps)
    if state.ndim != 1 or len(state) == 0:
        raise ValueError('register_state must be a non-empty one-dimensional array')
    if not np.all((state == 0) | (state == 1)) or not np.any(state):
        raise ValueError('register_state must be binary and not all zeros')
    if not taps or any(tap < 1 or tap > len(state) for tap in taps):
        raise ValueError('taps must be valid 1-based register positions')
    if length is None:
        length = 2 ** len(state) - 1
    if length <= 0:
        raise ValueError('length must be positive')

    output_bits = []

    for _ in range(length):
        # Output the rightmost bit.
        output_bit = state[-1]
        output_bits.append(output_bit)

        # Feedback is the XOR of tapped bits.
        feedback = 0
        for tap in taps:
            feedback ^= state[tap - 1]

        # Shift right and insert feedback at the left.
        state = np.concatenate(([feedback], state[:-1]))

    output_bits = np.asarray(output_bits, dtype=int)

    # Bipolar mapping: bit 0 -> +1, bit 1 -> -1.
    chips = np.where(output_bits == 0, 1, -1)
    return chips.astype(int)


def dsss_spread(bits, pn_chips):
    """
    Spread BPSK symbols with PN chips.

    For each bit, map 0 -> +1 and 1 -> -1, then multiply by the whole PN
    sequence. Output length is len(bits) * len(pn_chips).
    """
    bits = np.asarray(bits, dtype=int)
    pn_chips = _validate_pn_chips(pn_chips)
    if bits.ndim != 1 or not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits must be a one-dimensional binary array')

    # BPSK mapping: 0 -> +1, 1 -> -1.
    symbols = 1 - 2 * bits

    # Each symbol is multiplied by the whole PN sequence.
    spread_chips = symbols[:, np.newaxis] * pn_chips[np.newaxis, :]

    return spread_chips.reshape(-1)


def dsss_despread(received_chips, pn_chips):
    """
    Despread received chips by correlation with the same PN sequence.

    Returns:
        recovered bits after hard decision. Non-negative correlation -> bit 0.
    """
    received_chips = np.asarray(received_chips, dtype=float)
    pn_chips = _validate_pn_chips(pn_chips)
    if received_chips.ndim != 1 or len(received_chips) % len(pn_chips) != 0:
        raise ValueError('received_chips length must be a multiple of PN length')

    spreading_factor = len(pn_chips)

    # Reshape the received chips into one row per transmitted bit.
    chip_matrix = received_chips.reshape(-1, spreading_factor)

    # Correlate each row with the PN sequence.
    correlations = chip_matrix @ pn_chips

    # Decision rule:
    # correlation >= 0 -> symbol is closer to +PN -> bit 0
    # correlation < 0  -> symbol is closer to -PN -> bit 1
    recovered_bits = (correlations < 0).astype(int)

    return recovered_bits


def processing_gain_db(spreading_factor):
    """Return processing gain 10*log10(spreading_factor) in dB."""
    if spreading_factor <= 0:
        raise ValueError('spreading_factor must be positive')

    return 10 * np.log10(spreading_factor)


def despread_with_timing_offset(received_chips, pn_chips, max_offset):
    """Optional: search timing offset by maximum correlation magnitude."""
    if max_offset < 0:
        raise ValueError('max_offset must be non-negative')

    received_chips = np.asarray(received_chips, dtype=float)
    pn_chips = _validate_pn_chips(pn_chips)

    if received_chips.ndim != 1:
        raise ValueError('received_chips must be a one-dimensional array')

    spreading_factor = len(pn_chips)

    best_offset = 0
    best_metric = -np.inf
    best_recovered_bits = None

    for offset in range(max_offset + 1):
        # Remove the first "offset" chips and keep only complete DSSS symbols.
        remaining_length = len(received_chips) - offset

        if remaining_length < spreading_factor:
            continue

        usable_length = (remaining_length // spreading_factor) * spreading_factor
        candidate_chips = received_chips[offset: offset + usable_length]

        # Reshape into one row per symbol.
        chip_matrix = candidate_chips.reshape(-1, spreading_factor)

        # Correlate with the local PN sequence.
        correlations = chip_matrix @ pn_chips

        # A correct timing offset should produce larger correlation magnitude.
        metric = np.mean(np.abs(correlations))

        if metric > best_metric:
            best_metric = metric
            best_offset = offset

            # Hard decision:
            # correlation >= 0 -> bit 0
            # correlation < 0  -> bit 1
            best_recovered_bits = (correlations < 0).astype(int)

    if best_recovered_bits is None:
        raise ValueError('received_chips is too short for the given PN sequence')

    return best_recovered_bits


def _correlation_values(received_chips, pn_chips):
    matrix = np.asarray(received_chips, dtype=float).reshape(-1, len(pn_chips))
    return matrix @ np.asarray(pn_chips, dtype=float) / len(pn_chips)


def run_spread_spectrum_demo():
    """Run Part 2 demo and generate figures."""
    print('=' * 60)
    print('Part 2: DSSS 扩频通信实验')
    print('=' * 60)
    snr_db_values = np.array([-6, -3, 0, 3, 6, 9], dtype=float)

    try:
        pn_chips = generate_m_sequence([1, 1, 1, 0, 1], taps=[5, 2], length=31)
        bits = generate_bits(3000, seed=2026)
        unspread_ber = []
        dsss_ber = []

        for index, snr_db in enumerate(snr_db_values):
            symbols = bpsk_modulate(bits)
            unspread_rx = add_narrowband_interference(symbols, amplitude=0.8, frequency=0.11)
            unspread_rx = add_awgn(unspread_rx, snr_db, seed=100 + index)
            unspread_ber.append(calculate_ber(bits, bpsk_demodulate(unspread_rx)))

            chips = dsss_spread(bits, pn_chips)
            rx_chips = add_narrowband_interference(chips, amplitude=0.8, frequency=0.11)
            rx_chips = add_awgn(rx_chips, snr_db, seed=200 + index)
            recovered = dsss_despread(rx_chips, pn_chips)
            dsss_ber.append(calculate_ber(bits, recovered))

        plot_ber_curve(
            snr_db_values,
            {'未扩频': unspread_ber, f'DSSS(N={len(pn_chips)})': dsss_ber},
            '窄带干扰下 DSSS 扩频前后 BER 对比',
            'dsss_ber_curve.png',
        )

        demo_bits = generate_bits(120, seed=77)
        demo_chips = dsss_spread(demo_bits, pn_chips)
        demo_rx = add_narrowband_interference(demo_chips, amplitude=0.8, frequency=0.11)
        demo_rx = add_awgn(demo_rx, 0, seed=88)
        correlations = _correlation_values(demo_rx, pn_chips)
        plot_correlation_snapshot(correlations, 'dsss_correlation_snapshot.png')

        print(f'[OK] 处理增益: {processing_gain_db(len(pn_chips)):.2f} dB')
        print('[OK] 已生成 results/dsss_ber_curve.png')
        print('[OK] 已生成 results/dsss_correlation_snapshot.png')
    except NotImplementedError as error:
        print(f'[WAIT] 尚未完成核心函数: {error}')
    except Exception as error:
        print(f'[FAIL] Part 2 运行失败: {error}')


if __name__ == '__main__':
    run_spread_spectrum_demo()
