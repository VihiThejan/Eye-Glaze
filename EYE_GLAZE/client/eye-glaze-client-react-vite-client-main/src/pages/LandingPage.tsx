// LandingPage component
import { Link } from 'react-router-dom';
import { ArrowRight, Eye, Shield, TrendingUp, Zap, CheckCircle, Star, Users, Award } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export function LandingPage() {
  const features = [
    {
      icon: Eye,
      title: 'Advanced Eye Analysis',
      description: 'State-of-the-art AI algorithms analyze micro-expressions and eye patterns to detect stress levels with 95% accuracy.',
    },
    {
      icon: Shield,
      title: 'Privacy & Security',
      description: 'Enterprise-grade encryption ensures your biometric data is processed securely and never stored without explicit consent.',
    },
    {
      icon: TrendingUp,
      title: 'Comprehensive Analytics',
      description: 'Track your wellness journey with detailed insights, trends, and personalized recommendations over time.',
    },
    {
      icon: Zap,
      title: 'Real-time Processing',
      description: 'Get instant stress analysis results within seconds, powered by our cloud-based AI infrastructure.',
    },
  ];

  const testimonials = [
    {
      name: 'Dr. Sarah Chen',
      role: 'Wellness Specialist',
      content: 'Eye Glaze has revolutionized how we monitor patient stress levels. The accuracy is remarkable.',
      rating: 5,
    },
    {
      name: 'Michael Rodriguez',
      role: 'Corporate Executive',
      content: 'This platform helped me identify stress patterns I never knew existed. Life-changing technology.',
      rating: 5,
    },
    {
      name: 'Emma Thompson',
      role: 'Mental Health Advocate',
      content: 'The insights provided are incredibly detailed and actionable. Highly recommend for anyone serious about wellness.',
      rating: 5,
    },
  ];

  const stats = [
    { value: '50K+', label: 'Active Users' },
    { value: '95%', label: 'Accuracy Rate' },
    { value: '1M+', label: 'Analyses Completed' },
    { value: '24/7', label: 'Support Available' },
  ];

  return (
    <div className="min-h-screen pt-16">
      {/* Hero Section */}
      <section className="relative py-24 overflow-hidden">
        <div className="absolute inset-0 hero-gradient" />
        <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:50px_50px]" />
        
        <div className="container mx-auto px-4 relative">
          <div className="text-center max-w-5xl mx-auto">
            <Badge variant="secondary" className="mb-6 px-4 py-2 text-sm font-medium">
              <Star className="w-4 h-4 mr-2" />
              Trusted by 50,000+ users worldwide
            </Badge>
            
            <h1 className="text-6xl md:text-8xl font-bold mb-8 leading-tight">
              <span className="text-gradient">Eye Glaze</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground mb-6 font-medium">
              Revolutionary Stress Management Through
            </p>
            <p className="text-2xl md:text-3xl font-semibold text-primary mb-12">
              Advanced Eye Analysis Technology
            </p>
            
            <p className="text-lg text-muted-foreground mb-12 max-w-3xl mx-auto leading-relaxed">
              Harness the power of artificial intelligence to understand your stress levels through sophisticated 
              eye pattern analysis. Get personalized insights and evidence-based recommendations for optimal mental wellness.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-6 justify-center mb-16">
              <Button asChild size="lg" className="gradient-primary glow-primary text-lg px-8 py-6 h-auto">
                <Link to="/register">
                  Start Your Journey <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="text-lg px-8 py-6 h-auto border-primary/30 hover:border-primary">
                <Link to="/login">Sign In</Link>
              </Button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
              {stats.map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="text-3xl md:text-4xl font-bold text-primary mb-2">{stat.value}</div>
                  <div className="text-sm text-muted-foreground font-medium">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-muted/20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-20">
            <Badge variant="outline" className="mb-4 px-4 py-2">
              <Award className="w-4 h-4 mr-2" />
              Industry Leading Technology
            </Badge>
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Why Choose <span className="text-gradient">Eye Glaze</span>?
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Our cutting-edge platform combines medical research, artificial intelligence, and user-centric design 
              to deliver unparalleled stress analysis capabilities.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
            {features.map((feature, index) => (
              <Card key={index} className="glass-card hover:glow-primary transition-all duration-500 group">
                <CardHeader className="pb-4">
                  <div className="w-16 h-16 rounded-2xl gradient-primary flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                    <feature.icon className="h-8 w-8 text-white" />
                  </div>
                  <CardTitle className="text-2xl mb-3">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-base leading-relaxed text-muted-foreground">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24">
        <div className="container mx-auto px-4">
          <div className="text-center mb-20">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              How It <span className="text-gradient">Works</span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Experience the future of stress management in three simple steps. Our streamlined process 
              ensures accurate results with minimal effort.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-12 max-w-6xl mx-auto">
            {[
              {
                step: '01',
                title: 'Capture Eye Image',
                description: 'Use our guided camera interface to take a high-quality photo of your eye. Our system provides real-time feedback for optimal image quality.',
                icon: Eye,
              },
              {
                step: '02',
                title: 'AI-Powered Analysis',
                description: 'Our advanced machine learning algorithms analyze pupil dilation, eye movement patterns, and micro-expressions within seconds.',
                icon: Zap,
              },
              {
                step: '03',
                title: 'Personalized Insights',
                description: 'Receive detailed stress analysis with actionable recommendations, trend tracking, and wellness coaching tailored to your needs.',
                icon: TrendingUp,
              },
            ].map((item, index) => (
              <div key={index} className="text-center group">
                <div className="relative mb-8">
                  <div className="w-20 h-20 rounded-full gradient-primary glow-primary flex items-center justify-center text-white font-bold text-xl mb-6 mx-auto group-hover:scale-110 transition-transform duration-300">
                    {item.step}
                  </div>
                  <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center mx-auto">
                    <item.icon className="w-6 h-6 text-primary" />
                  </div>
                </div>
                <h3 className="text-2xl font-semibold mb-4">{item.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-24 bg-muted/20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-20">
            <Badge variant="outline" className="mb-4 px-4 py-2">
              <Users className="w-4 h-4 mr-2" />
              Trusted by Professionals
            </Badge>
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              What Our <span className="text-gradient">Users Say</span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Join thousands of satisfied users who have transformed their approach to stress management 
              with Eye Glaze's innovative technology.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {testimonials.map((testimonial, index) => (
              <Card key={index} className="glass-card">
                <CardHeader>
                  <div className="flex items-center space-x-1 mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} className="w-5 h-5 fill-primary text-primary" />
                    ))}
                  </div>
                  <CardDescription className="text-base leading-relaxed">
                    "{testimonial.content}"
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center space-x-3">
                    <div className="w-12 h-12 rounded-full gradient-primary flex items-center justify-center text-white font-semibold">
                      {testimonial.name.charAt(0)}
                    </div>
                    <div>
                      <div className="font-semibold">{testimonial.name}</div>
                      <div className="text-sm text-muted-foreground">{testimonial.role}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24">
        <div className="container mx-auto px-4">
          <Card className="glass-card max-w-4xl mx-auto text-center p-12">
            <CardHeader>
              <CardTitle className="text-4xl md:text-5xl font-bold mb-6">
                Ready to Transform Your <span className="text-gradient">Wellness Journey</span>?
              </CardTitle>
              <CardDescription className="text-xl leading-relaxed mb-8 max-w-2xl mx-auto">
                Join the revolution in stress management. Start your free trial today and discover 
                insights about your mental wellness that you never knew existed.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col sm:flex-row gap-6 justify-center mb-8">
                <Button asChild size="lg" className="gradient-primary glow-primary text-lg px-8 py-6 h-auto">
                  <Link to="/register">
                    Start Free Trial <ArrowRight className="ml-2 h-5 w-5" />
                  </Link>
                </Button>
                <Button asChild variant="outline" size="lg" className="text-lg px-8 py-6 h-auto border-primary/30 hover:border-primary">
                  <Link to="/login">Sign In</Link>
                </Button>
              </div>
              
              <div className="flex items-center justify-center space-x-6 text-sm text-muted-foreground">
                <div className="flex items-center">
                  <CheckCircle className="w-4 h-4 text-primary mr-2" />
                  No credit card required
                </div>
                <div className="flex items-center">
                  <CheckCircle className="w-4 h-4 text-primary mr-2" />
                  14-day free trial
                </div>
                <div className="flex items-center">
                  <CheckCircle className="w-4 h-4 text-primary mr-2" />
                  Cancel anytime
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  );
}