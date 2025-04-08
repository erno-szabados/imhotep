package com.esgdev.imhotep;

import org.jocl.*;

import java.util.logging.*;

public class Main {
    public static final Logger logger = Logger.getLogger(Main.class.getName());

    public static void main(String[] args) {
        JoclProcessor processor = new JoclProcessor();

        String kernelSource =
                "__kernel void add(__global const float *a, __global const float *b, __global float *c) {\n" +
                        "    int i = get_global_id(0);\n" +
                        "    c[i] = a[i] + b[i];\n" +
                        "}\n";

        int n = 10;
        float[] a = new float[n];
        float[] b = new float[n];
        float[] c = new float[n];

        for (int i = 0; i < n; i++) {
            a[i] = i;
            b[i] = n - i;
        }

        processor.createKernel(kernelSource, "add");

        // Create memory buffers for the input and output data
        cl_mem memA = processor.createReadOnlyBuffer(a);
        cl_mem memB = processor.createReadOnlyBuffer(b);
        cl_mem memC = processor.createWriteOnlyBuffer(Sizeof.cl_float * n);

        // Set the arguments for the kernel
        processor.setKernelArg(0, Sizeof.cl_mem, Pointer.to(memA));
        processor.setKernelArg(1, Sizeof.cl_mem, Pointer.to(memB));
        processor.setKernelArg(2, Sizeof.cl_mem, Pointer.to(memC));

        // Execute the kernel
        processor.executeKernel(n);

        // Read the results from the output buffer
        processor.readResults(memC, c, Sizeof.cl_float);

        for (float value : c) {
            System.out.print(value + " ");
        }
        System.out.println();

        // Release memory objects and OpenCL resources
        processor.releaseResources(memA, memB, memC);
    }
}