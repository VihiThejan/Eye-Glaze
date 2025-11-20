import { useState, useEffect } from 'react';
import { CheckCircle, Target, Zap, Calendar, BarChart, Leaf, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import { User } from '@/types';
import { Separator } from '@/components/ui/separator';

// Default recommendations are now directly in fetchRecommendations function

interface AnalysisStats {
  totalAnalysesLastWeek: number;
  stressDetectedCount: number;
  stressPercentage: number;
}

export function StressRecommendations() {
  const [user, setUser] = useState<User | null>(null);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [lifestyleAdjustments, setLifestyleAdjustments] = useState<string[]>([]);
  const [assessment, setAssessment] = useState<string>('');
  const [stats, setStats] = useState<AnalysisStats | null>(null);
  const [isLoading, setIsLoading] = useState(false); // Start with not loading
  const [error, setError] = useState<string | null>(null);
  const [hasGeneratedRecommendations, setHasGeneratedRecommendations] = useState(false);
  
  // Get user from localStorage
  useEffect(() => {
    const getUserFromStorage = (): User | null => {
      const storedUser = localStorage.getItem('eyeGlazeUser');
      return storedUser ? JSON.parse(storedUser) : null;
    };
    
    setUser(getUserFromStorage());
  }, []);
  
  // Define the fetch recommendations function outside useEffect so we can call it on demand
  const fetchRecommendations = async (forceGenerate = false) => {
    if (!user?.email) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Using the correct endpoint as per API docs: /api/recommendations/user/:username
      // Add a query parameter for force generation when button is clicked
      const url = `http://localhost:5176/api/recommendations/user/${user.email}${forceGenerate ? '?generate=true' : ''}`;
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch recommendations: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.status === "success" && result.data) {
        // Set stats if available
        if (result.data.stats) {
          setStats(result.data.stats);
        }
        
        // Set recommendations data
        if (result.data.recommendations) {
          // Set assessment
          if (result.data.recommendations.assessment) {
            setAssessment(result.data.recommendations.assessment);
          }
          
          // Set recommendations
          if (Array.isArray(result.data.recommendations.recommendations)) {
            setRecommendations(result.data.recommendations.recommendations);
          }
          
          // Set lifestyle adjustments
          if (Array.isArray(result.data.recommendations.lifestyleAdjustments)) {
            setLifestyleAdjustments(result.data.recommendations.lifestyleAdjustments);
          }
        }
        
        // Show success message when recommendations are successfully generated
        toast.success("Recommendations generated successfully!");
        setHasGeneratedRecommendations(true);
      }
    } catch (err) {
      console.error("Error fetching recommendations:", err);
      setError(err instanceof Error ? err.message : "Failed to load recommendations");
      
      // Use default recommendations if there's an error
      if (recommendations.length === 0) {
        setRecommendations([
          'Take a 5-minute mindfulness break to reset your nervous system',
          'Practice the 4-7-8 breathing technique for immediate stress relief',
          'Reduce screen exposure for the next 30 minutes to rest your eyes',
          'Consider a brief walk outdoors to boost natural mood regulators',
          'Stay hydrated - dehydration can amplify stress responses',
        ]);
      }
    } finally {
      setIsLoading(false);
    }
  };
  
  // Just load the user from localStorage, but don't automatically fetch recommendations
  // We'll only fetch recommendations when the user clicks the button
  
  const startGuidedSession = () => {
    toast.info("Guided session feature coming soon!");
  };
  
  const scheduleReminder = () => {
    toast.info("Reminder scheduling feature coming soon!");
  };
  
  // Helper function to clean recommendation text by removing markdown styling markers
  const cleanText = (text: string): string => {
    if (!text) return '';
    
    // Based on the screenshot, we're seeing patterns like:
    // "**Practice the 20-20-20 rule**:" and "**Practice the 20-20-20 rule**:"
    
    // Remove all instances of ** from the text 
    return text.replace(/\*\*/g, '');
  };
  
  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2 text-2xl">
          <Target className="h-6 w-6" />
          <span>Personalized Recommendations</span>
        </CardTitle>
        <CardDescription className="text-base">
          Evidence-based actions to help optimize your current stress level
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center p-12">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary"></div>
          </div>
        ) : !hasGeneratedRecommendations ? (
          <div className="text-center py-12">
            <div className="mb-6">
              <Target className="h-16 w-16 mx-auto text-primary opacity-50" />
              <h3 className="text-xl font-medium mt-4 mb-2">No Recommendations Generated Yet</h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                Generate personalized recommendations based on your recent eye scan analyses and stress patterns.
              </p>
            </div>
            
            <Button 
              className="gradient-primary glow-primary mx-auto" 
              size="lg"
              onClick={() => fetchRecommendations(true)} 
            >
              <RefreshCw className="w-5 h-5 mr-2" />
              Generate Recommendations
            </Button>
          </div>
        ) : (
          <>
            {error && (
              <Alert className="border-destructive/20 bg-destructive/5 mb-6">
                <AlertDescription className="text-base">
                  {error} - There was a problem generating recommendations.
                </AlertDescription>
              </Alert>
            )}
          
            <div className="space-y-6">
            {/* Assessment Section */}
            {assessment && (
              <div className="assessment-container bg-card/50 p-5 rounded-lg border shadow-sm">
                <h4 className="text-lg font-medium mb-2 flex items-center">
                  <BarChart className="h-5 w-5 mr-2 text-primary" />
                  Assessment Summary
                </h4>
                <p className="text-muted-foreground">{cleanText(assessment)}</p>
                
                {stats && (
                  <div className="stats-container mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
                    {Object.entries(stats).map(([key, value]) => (
                      <div key={key} className="stat-item bg-background/50 p-3 rounded-md">
                        <p className="text-sm text-muted-foreground capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</p>
                        <p className="text-lg font-medium">{typeof value === 'number' ? value.toFixed(1) : value}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            
            {/* Recommendations Section */}
            {recommendations.length > 0 && (
              <div>
                <h4 className="text-lg font-medium mb-2 flex items-center">
                  <CheckCircle className="h-5 w-5 mr-2 text-primary" />
                  Recommended Actions
                </h4>
                <div className="grid gap-4">
                  {recommendations.map((recommendation, index) => (
                    <Alert key={index} className="border-primary/20 bg-primary/5">
                      <CheckCircle className="h-5 w-5 text-primary" />
                      <AlertDescription className="text-base ml-2">
                        {cleanText(recommendation)}
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              </div>
            )}
            
            {/* Lifestyle Adjustments Section */}
            {lifestyleAdjustments.length > 0 && (
              <div>
                <Separator className="my-4" />
                <h4 className="text-lg font-medium mb-2 flex items-center">
                  <Leaf className="h-5 w-5 mr-2 text-primary" />
                  Lifestyle Adjustments
                </h4>
                <div className="grid gap-3">
                  {lifestyleAdjustments.map((adjustment, index) => (
                    <Alert key={index} className="border-primary/20 bg-secondary/5">
                      <Leaf className="h-5 w-5 text-secondary" />
                      <AlertDescription className="text-base ml-2">
                        {cleanText(adjustment)}
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              </div>
            )}
          </div>
          </>
        )}
        {hasGeneratedRecommendations && (
          <div className="mt-6 flex flex-wrap gap-4">
            <Button 
              className="gradient-primary glow-primary" 
              onClick={() => fetchRecommendations(true)} 
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <div className="animate-spin mr-2 h-4 w-4 border-2 border-t-transparent rounded-full"></div>
                  Generating...
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh Recommendations
                </>
              )}
            </Button>
            <Button className="gradient-secondary" onClick={startGuidedSession}>
              <Zap className="w-4 h-4 mr-2" />
              Start Guided Session
            </Button>
            <Button variant="outline" onClick={scheduleReminder}>
              <Calendar className="w-4 h-4 mr-2" />
              Schedule Reminder
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}