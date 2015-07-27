import java.io.File;


public class DirectoryUtils {
	public static String getFilename(String directory, String pattern) {
		File dir = new File(directory);
		
		
		for (File file : dir.listFiles()) {
			if (file.getName().matches(pattern)) {
				return file.getAbsolutePath();
			}
		}
		
		return null;
	}
}
