'\nPure-Python Implementation of the AES block-cipher.\n\nBenchmark AES in CTR mode using the pyaes module.\n'
import pyperf
import pyaes

CLEARTEXT: bytes = (b'This is a test. What could possibly go wrong? ' * 500)
KEY: bytes = b'\xa1\xf6%\x8c\x87}_\xcd\x89dHE8\xbf\xc9,'

def bench_pyaes(loops: int) -> float:
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for loops in range_it:
        aes = pyaes.AESModeOfOperationCTR(KEY)
        ciphertext = aes.encrypt(CLEARTEXT)
        aes = pyaes.AESModeOfOperationCTR(KEY)
        plaintext = aes.decrypt(ciphertext)
        aes = None
    dt = (pyperf.perf_counter() - t0)
    if (plaintext != CLEARTEXT):
        raise Exception('decrypt error!')
    return dt

if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Pure-Python Implementation of the AES block-cipher'
    runner.bench_time_func('crypto_pyaes', bench_pyaes)
