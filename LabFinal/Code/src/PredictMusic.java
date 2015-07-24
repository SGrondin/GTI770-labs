import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Vote;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


public class PredictMusic {

	
	public static void main(String[] args) throws Exception {
		String step = "build";
		String inBuildFolder = "../Sample/";
		String inEvalFolder = "../Sample/";
		String modelFolder = "../Model/";
		ObjectOutputStream oos;
		ObjectInputStream ois;
		double[] result;
		Instances combined;
		
		String folder = "build".equals(step) ? inBuildFolder : inEvalFolder;
		Instances jmirmfccs = DataSource.read(folder + "msd-jmirmfccs_dev.arff");
		Instances marsyas = DataSource.read(folder + "msd-marsyas_dev_new.arff");
		Instances ssd = DataSource.read(folder + "msd-ssd_dev.arff");
		Instances rh = DataSource.read(folder + "msd-rh_dev_new.arff");
		
		// On enlève les attributes textes qui ne sont pas pertinents //
		jmirmfccs = removeIdentifier(jmirmfccs);
		marsyas = removeIdentifier(marsyas);
		ssd = removeIdentifier(ssd);
		rh = removeIdentifier(rh);
		
		if ("build".equals(step)) {
			// Modele pour Bayes
			Classifier bayes = strategyBayes(jmirmfccs, marsyas, ssd);
			oos = new ObjectOutputStream(new FileOutputStream(modelFolder + "bayes.model"));
			oos.writeObject(bayes);
			oos.flush();
			oos.close();
			
			// Modele pour KNN
			Classifier knn = strategyKNN(jmirmfccs, marsyas, ssd);
			oos = new ObjectOutputStream(new FileOutputStream(modelFolder + "knn.model"));
			oos.writeObject(knn);
			oos.flush();
			oos.close();
		}
		
		if ("evaluate".equals(step)) {
			// Les trois premiers modeles sont utilisent ensemble pour 2 classificateurs
			combined = mergeInstances(jmirmfccs, marsyas, ssd);
			
			ois = new ObjectInputStream(new FileInputStream(modelFolder + "bayes.model"));
			Classifier classifier = (Classifier) ois.readObject();
			
			result = classifyInstances(classifier, combined);
			String a = combined.attribute(combined.classIndex()).value((int) result[0]);
			String e = combined.firstInstance().stringValue(combined.classIndex());
			
			System.out.println("Predicted : " + a + " Expected : " + e);
		}
		
	}
	
	private static double[] classifyInstances(Classifier classifier, Instances inst) throws Exception {
		double[] result = new double[inst.numInstances()];
		
		for (int i=0; i<result.length; i++) {
			Instance ins = inst.instance(i);
			result[i] = classifier.classifyInstance(ins);
		}
		
		return result;
	}
	
	private static Classifier strategyKNN(Instances... set) throws Exception {
		Classifier[] classifiers = new Classifier[set.length];
		
		for (int i=0; i<set.length; i++) {
			Instances inst = set[i];
			IBk ibk = new IBk();
			ibk.setOptions(new String[]{ "-K", "3", "-W", "0", "-A", "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" });
			ibk.buildClassifier(inst);
			classifiers[i] = ibk;
		}
		
		Vote vote = new Vote();
		vote.setOptions(new String[] { "-R", "MED" });
		vote.setClassifiers(classifiers);
		
		return vote;
	}
	
	private static Classifier strategyBayes(Instances... set) throws Exception {
		Classifier[] classifiers = new Classifier[set.length];
		
		for (int i=0; i<set.length; i++) {
			Instances inst = set[i];
			NaiveBayes bayes = new NaiveBayes();
			bayes.setOptions(new String[]{ "-D" });
			bayes.buildClassifier(inst);
			classifiers[i] = bayes;
		}
		
		Vote vote = new Vote();
		vote.setOptions(new String[] { "-R", "MED" });
		vote.setClassifiers(classifiers);
		
		return vote;
	}

	private static Instances mergeInstances(Instances... set) throws Exception {
		Instances result = set[0];
		
		for (int i=1; i<set.length; i++) {
			// On enleve l'attribut de classe pour eviter les erreurs d'attribut
			// non unique.
			Remove remove = new Remove();
			remove.setOptions(new String[]{ "-R", set[i].numAttributes() + "" });
			remove.setInputFormat(set[i]);
			
			Instances instNew = Filter.useFilter(set[i], remove);
			
			// On merge avec l'ensemble que l'on a deja.
			result = Instances.mergeInstances(instNew, result);
		}
		
		result.setClassIndex(result.numAttributes() - 1);
		return result;
	}
	
	private static Instances removeIdentifier(Instances inst) throws Exception {
		Remove remove = new Remove();
		remove.setOptions(new String[]{ "-R", "1,2" });
		remove.setInputFormat(inst);
		
		Instances instNew = Filter.useFilter(inst, remove);
		instNew.setClassIndex(instNew.numAttributes() - 1);
		
		return instNew;
	}
}
