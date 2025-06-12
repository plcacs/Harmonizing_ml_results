'''
Pure-Python Implementation of the AES block-cipher.

Benchmark AES in CTR mode using the pyaes module.
'''
import pyperf
import pyaes

CLEARTEXT: bytes = (b'This is a test. What could possibly go wrong? ' * 500)
KEY: bytes = b'\xa1\xf6%\x8c\x87}_\xcd\x89dHE8\xbf\xc9,'

def bench_pyaes(loops: int) -> float:
    range_it: range = range(loops)
    t0: float = pyperf.perf_counter()
    plaintext: bytes = b''
    for _ in range_it:
        aes: pyaes.AESModeOfOperationCTR = pyaes.AESModeOfOperationCTR(KEY)
        ciphertext: bytes = aes.encrypt(CLEARTEXT)
        aes = pyaes.AESModeOfOperationCTR(KEY)
        plaintext = aes.decrypt(ciphertext)
        aes = None
    dt: float = pyperf.perf_counter() - t0
    if plaintext != CLEARTEXT:
        raise Exception('decrypt error!')
    return dt

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Pure-Python Implementation of the AES block-cipher'
    runner.bench_time_func('crypto_pyaes', bench_pyaes)
