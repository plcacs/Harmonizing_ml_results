"""
This module contains functions that allow sending type-checked `RunInput` data
to flows at runtime. Flows can send back responses, establishing two-way
channels with senders. These functions are particularly useful for systems that
require ongoing data transfer or need to react to input quickly.
real-time interaction and efficient data handling. It's designed to facilitate
dynamic communication within distributed or microservices-oriented systems,
making it ideal for scenarios requiring continuous data synchronization and
processing. It's particularly useful for systems that require ongoing data
input and output.

The following is an example of two flows. One sends a random number to the
other and waits for a response. The other receives the number, squares it, and
sends the result back. The sender flow then prints the result.

Sender flow:

