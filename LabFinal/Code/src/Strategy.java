import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Vote;
import weka.core.Instances;


public class Strategy {
	public static Classifier strategyKNN(Instances... set) throws Exception {
		Classifier[] classifiers = new Classifier[set.length];
		
		for (int i=0; i<set.length; i++) {
			Instances inst = set[i];
			IBk ibk = new IBk();
			ibk.setOptions(new String[]{ "-K", "3", "-W", "0", "-A", "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" });
			
			Classifier classifier = new AttributeClassifier(ibk);
			classifier.buildClassifier(inst);
			classifiers[i] = classifier;
		}
		
		Vote vote = new Vote();
		vote.setClassifiers(classifiers);
		return vote;
	}
	
	public static Classifier strategyBayes(Instances... set) throws Exception {
		Classifier[] classifiers = new Classifier[set.length];
		
		for (int i=0; i<set.length; i++) {
			Instances inst = set[i];
			NaiveBayes bayes = new NaiveBayes();
			bayes.setOptions(new String[]{ "-D" });
			
			Classifier classifier = new AttributeClassifier(bayes);
			classifier.buildClassifier(inst);
			classifiers[i] = classifier;
		}
		
		Vote vote = new Vote();
		vote.setClassifiers(classifiers);
		return vote;
	}
}
