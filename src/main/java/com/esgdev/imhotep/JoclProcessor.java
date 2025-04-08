package com.esgdev.imhotep;

import org.jocl.*;

import java.nio.ByteBuffer;
import java.util.logging.Logger;

import static org.jocl.CL.*;

/**
 * JoclProcessor is a class that initializes OpenCL, creates a kernel, and manages memory buffers
 * for OpenCL operations. It provides methods to prepare data, execute the kernel, read results,
 * and release resources.
 */
public class JoclProcessor {

    public static final Logger logger = Logger.getLogger(JoclProcessor.class.getName());

    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_program program;
    private cl_kernel kernel;
    private cl_device_id device;

    public JoclProcessor() {
        enumeratePlatforms();
        initOpenCL();
    }

    public cl_context getContext() {
        return context;
    }

    private void initOpenCL() {
        int platformIndex = 0;
        int deviceIndex = 0;

        cl_platform_id[] platforms = new cl_platform_id[1];
        // Get the first platform
        clGetPlatformIDs(1, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, devices, null);
        this.device = devices[deviceIndex];

        this.context = clCreateContext(null, 1, new cl_device_id[]{this.device}, null, null, null);
        cl_queue_properties properties = new cl_queue_properties();
        this.commandQueue = clCreateCommandQueueWithProperties(context, device, properties, null);

    }

    private void enumeratePlatforms() {
        int[] numPlatforms = new int[1];

        // Get the number of platforms
        int ret = clGetPlatformIDs(0, null, numPlatforms);
        if (ret != CL_SUCCESS) {
            logger.severe("Error getting number of platforms: " + ret);
            return;
        }
        logger.info("Number of platforms: " + numPlatforms[0]);

        // Get the platform IDs
        cl_platform_id[] platform = new cl_platform_id[numPlatforms[0]];
        ret = clGetPlatformIDs(numPlatforms[0], platform, null);
        if (ret != CL_SUCCESS) {
            logger.severe("Error getting platform IDs: " + ret);
            return;
        }

        // Log platform information
        for (int i = 0; i < numPlatforms[0]; i++) {
            logger.info(getPlatformInfoString(platform[i]));
            enumerateDevices(platform[i]);
        }
    }

    private void enumerateDevices(cl_platform_id platform) {
        int[] numDevices = new int[1];

        // Get the number of devices
        int ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevices);
        if (ret != CL_SUCCESS) {
            logger.severe("Error getting number of devices: " + ret);
            return;
        }
        logger.info("Number of devices: " + numDevices[0]);

        // Get the device IDs
        cl_device_id[] devices = new cl_device_id[numDevices[0]];
        ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices[0], devices, null);
        if (ret != CL_SUCCESS) {
            logger.severe("Error getting device IDs: " + ret);
            return;
        }

        // Log device information
        for (int i = 0; i < numDevices[0]; i++) {
            logger.info(getDeviceInfoString(devices[i]));
        }
    }

    private static String getDeviceInfoString(cl_device_id device) {
        StringBuilder sb = new StringBuilder();

        String[] attributeNames = {
                "CL_DEVICE_NAME",
                "CL_DEVICE_TYPE",
                "CL_DEVICE_VENDOR",
                "CL_DEVICE_VERSION",
                "CL_DRIVER_VERSION",
                "CL_DEVICE_MAX_COMPUTE_UNITS",
                "CL_DEVICE_MAX_WORK_ITEM_SIZES"
        };

        int[] attributeValues = {
                CL_DEVICE_NAME,
                CL_DEVICE_TYPE,
                CL_DEVICE_VENDOR,
                CL_DEVICE_VERSION,
                CL_DRIVER_VERSION,
                CL_DEVICE_MAX_COMPUTE_UNITS,
                CL_DEVICE_MAX_WORK_ITEM_SIZES
        };

        for (int i = 0; i < attributeNames.length; i++) {
            String attributeName = attributeNames[i];
            int attributeValue = attributeValues[i];

            long[] size = new long[1];
            clGetDeviceInfo(device, attributeValue, 0, null, size);
            byte[] buffer = new byte[(int) size[0]];
            clGetDeviceInfo(device, attributeValue, buffer.length, Pointer.to(buffer), null);
            if (attributeValue == CL_DEVICE_TYPE) {
                long clDeviceType = ByteBuffer.wrap(buffer).order(java.nio.ByteOrder.nativeOrder()).getLong(0);
                sb.append(attributeName)
                        .append(": ")
                        .append(stringFor_cl_device_type(clDeviceType))
                        .append("\n");
            } else if (attributeValue == CL_DEVICE_MAX_COMPUTE_UNITS) {
                int maxComputeUnits = ByteBuffer.wrap(buffer).order(java.nio.ByteOrder.nativeOrder()).getInt(0);
                sb.append(attributeName)
                        .append(": ")
                        .append(maxComputeUnits)
                        .append("\n");
            } else if (attributeValue == CL_DEVICE_MAX_WORK_ITEM_SIZES) {
                int maxWorkItemSizes = ByteBuffer.wrap(buffer).order(java.nio.ByteOrder.nativeOrder()).getInt(0);
                sb.append(attributeName)
                        .append(": ")
                        .append(maxWorkItemSizes)
                        .append("\n");
            }
            else {
                sb.append(attributeName)
                        .append(": ")
                        .append(new String(buffer, 0, buffer.length - 1))
                        .append("\n");
            }
        }

        return sb.toString();
    }

    private static String getDeviceTypeString(long deviceType) {
        if ((deviceType & CL_DEVICE_TYPE_CPU) != 0) return "CPU";
        if ((deviceType & CL_DEVICE_TYPE_GPU) != 0) return "GPU";
        if ((deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0) return "ACCELERATOR";
        if ((deviceType & CL_DEVICE_TYPE_DEFAULT) != 0) return "DEFAULT";
        return "UNKNOWN";
    }

    private static String getPlatformInfoString(cl_platform_id platform) {
        StringBuilder sb = new StringBuilder();

        String[] attributeNames = {
                "CL_PLATFORM_PROFILE",
                "CL_PLATFORM_VERSION",
                "CL_PLATFORM_NAME",
                "CL_PLATFORM_VENDOR",
                "CL_PLATFORM_EXTENSIONS"
        };

        int[] attributeValues = {
                CL_PLATFORM_PROFILE,
                CL_PLATFORM_VERSION,
                CL_PLATFORM_NAME,
                CL_PLATFORM_VENDOR,
                CL_PLATFORM_EXTENSIONS
        };

        for (int i = 0; i < attributeNames.length; i++) {
            String attributeName = attributeNames[i];
            int attributeValue = attributeValues[i];

            long[] size = new long[1];
            clGetPlatformInfo(platform, attributeValue, 0, null, size);
            byte[] buffer = new byte[(int) size[0]];
            clGetPlatformInfo(platform, attributeValue, buffer.length, Pointer.to(buffer), null);

            sb.append(attributeName)
                    .append(": ")
                    .append(new String(buffer, 0, buffer.length - 1))
                    .append("\n");
        }

        return sb.toString();
    }

    public void createKernel(String kernelSource, String kernelName) {
        this.program = clCreateProgramWithSource(this.context, 1, new String[]{kernelSource}, null, null);
        clBuildProgram(this.program, 1, new cl_device_id[]{this.device}, null, null, null);
        this.kernel = clCreateKernel(this.program, kernelName, null);
    }

    public cl_mem createReadOnlyBuffer(float[] data) {
        int sizeof = Sizeof.cl_float;
        long size = (long) sizeof * data.length;
        float[] temp = new float[data.length];
        System.arraycopy(data, 0, temp, 0, data.length);
        return clCreateBuffer(this.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, Pointer.to(temp), null);
    }

    public cl_mem createReadOnlyBuffer(int[] data) {
        int sizeof = Sizeof.cl_int;
        long size = (long) sizeof * data.length;
        int[] temp = new int[data.length];
        System.arraycopy(data, 0, temp, 0, data.length);
        return clCreateBuffer(this.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, Pointer.to(temp), null);
    }

    public cl_mem createReadOnlyBuffer(double[] data) {
        int sizeof = Sizeof.cl_double;
        long size = (long) sizeof * data.length;
        double[] temp = new double[data.length];
        System.arraycopy(data, 0, temp, 0, data.length);
        return clCreateBuffer(this.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, Pointer.to(temp), null);
    }

    public cl_mem createWriteOnlyBuffer(long size) {
        return clCreateBuffer(this.context, CL_MEM_WRITE_ONLY, size, null, null);
    }

    public void setKernelArg(int argIndex, long size, Pointer data) {
        clSetKernelArg(this.kernel, argIndex, size, data);
    }

    public void executeKernel(int globalWorkSize) {
        long[] globalWorkSizeArr = new long[]{globalWorkSize};
        clEnqueueNDRangeKernel(this.commandQueue, this.kernel, 1, null, globalWorkSizeArr, null, 0, null, null);
    }

    public <T> void readResults(cl_mem mem, float[] output, int sizeof) {
        clEnqueueReadBuffer(this.commandQueue, mem, CL_TRUE, 0, (long) sizeof * output.length, Pointer.to(output), 0, null, null);
    }

    public void releaseResources(cl_mem... memObjects) {
        for (cl_mem memObject : memObjects) {
            clReleaseMemObject(memObject);
        }
        clReleaseKernel(this.kernel);
        clReleaseProgram(this.program);
        clReleaseCommandQueue(this.commandQueue);
        clReleaseContext(this.context);
    }
}